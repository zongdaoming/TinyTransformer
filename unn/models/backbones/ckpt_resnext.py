import logging

import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

from ..initializer import initialize_from_cfg
from ...extensions import DeformableConvInOne
from ..normalize import build_norm_layer, parse_deprecated_bn_style
from ...utils.checkpoint import fully_checkpoint_sequential
from ..components import build_conv_norm

logger = logging.getLogger('global')

__all__ = [
    'resnext_101_32x4d', 'resnext_101_32x8d', 'resnext_101_64x4d', 'resnext_101_64x8d', 'resnext_152_32x4d',
    'resnext_152_32x8d', 'resnext_152_64x4d', 'resnext_152_64x8d'
]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def init_weights(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, std)


class ResNeXtBottleneck(nn.Module):
    """RexNeXt bottleneck type C"""

    expansion = 2

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 base_width,
                 widen_factor,
                 deformable=False,
                 normalize={'type': 'solo_bn'}):
        """
        Arguments:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()

        key_conv = nn.Conv2d
        if deformable:
            key_conv = DeformableConvInOne

        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)

        norm_reduce_name, norm_reduce = build_norm_layer(D, normalize)
        norm_name, norm = build_norm_layer(D, normalize)
        norm_expand_name, norm_expand = build_norm_layer(out_channels, normalize)

        self.norm_reduce_name = norm_reduce_name + '_reduce'
        self.norm_name = norm_name
        self.norm_expand_name = norm_expand_name + '_expand'

        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.add_module(self.norm_reduce_name, norm_reduce)
        self.conv_conv = key_conv(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.add_module(self.norm_name, norm)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.add_module(self.norm_expand_name, norm_expand)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            shortcut_norm_name, norm = build_norm_layer(out_channels, normalize)
            self.shortcut.add_module(
                'shortcut_conv',
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_' + shortcut_norm_name, norm)

    @property
    def norm_reduce(self):
        return getattr(self, self.norm_reduce_name)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    @property
    def norm_expand(self):
        return getattr(self, self.norm_expand_name)

    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.norm_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.norm(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.norm_expand(bottleneck)
        residual = self.shortcut(x)
        return F.relu(residual + bottleneck, inplace=True)


class ResNeXt(nn.Module):
    """
    """
    def __init__(self,
                 block,
                 layers,
                 cardinality,
                 base_width,
                 out_layers,
                 out_strides,
                 bn=None,
                 normalize={'type': 'freeze_bn'},
                 frozen_layers=None,
                 deformable=None,
                 checkpoint=False,
                 layer_deform=None,
                 initializer=None):
        """
        Arguments:
            - frozen_layers (:obj:`list` of :obj:`int`): index of frozen layers, [0,4]
            - out_layers (:obj:`list` of :obj:`int`): index of output layers, [0,4]
            - out_strides (:obj:`list` of :obj:`int`): stride of outputs
            - layer_deform (:obj:`list` of (:obj:`str` or None): ``DCN`` setting for each layer. See note below
            - deformable (:obj:`bool`): Deprecated (see layer_deform). Use DCN or not. When ``True``, it's equivalent
              to ``layer_deform=[False, False, all, all, all]``
            - bn (:obj:`dict`): Deprecated (see normalize). Config of BatchNorm.
            - normalize (:obj:`dict`): Config of Normalization Layer (see Configuration#Normalization).
            - initializer (:obj:`dict`): Config of initilizaiton
        """
        super(ResNeXt, self).__init__()

        if bn is not None:
            normalize = parse_deprecated_bn_style(bn)

        frozen_layers = [] if frozen_layers is None else frozen_layers
        if frozen_layers is not None and len(frozen_layers) > 0:
            assert min(frozen_layers) >= 0, frozen_layers
            assert max(frozen_layers) <= 4, frozen_layers
        assert min(out_layers) >= 0, out_layers
        assert max(out_layers) <= 4, out_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.frozen_layers = frozen_layers
        midplanes = [64, 256, 512, 1024, 2048]
        self.out_planes = [midplanes[i] for i in self.out_layers]
        self.segments = self.get_segments(checkpoint)

        if layer_deform is None:
            logger.warning("Argument `deformable` will be deprecated, pls use layer_deform instead")
            if deformable:
                layer_deform = [None, None, 'last', 'last', 'last']
            else:
                layer_deform = [None] * 5
        assert len(layer_deform) == 5, layer_deform
        assert not layer_deform[0] and not layer_deform[1]

        self.norm1_name, norm1 = build_norm_layer(64, normalize, 1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 should not use deformable
        self.layer1 = self._make_layer(block, 64, 256, layers[0], 1, cardinality, base_width,
                                       normalize=normalize)
        self.layer2 = self._make_layer(
            block, 256, 512, layers[1], 2, cardinality, base_width, deformable=layer_deform[2],
            normalize=normalize)
        self.layer3 = self._make_layer(
            block, 512, 1024, layers[2], 2, cardinality, base_width, deformable=layer_deform[3],
            normalize=normalize)
        self.layer4 = self._make_layer(
            block, 1024, 2048, layers[3], 2, cardinality, base_width, deformable=layer_deform[4],
            normalize=normalize)

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params
        # to optimizer
        self.freeze_layer()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def layer0(self):
        return nn.Sequential(self.conv1, self.norm1, self.relu, self.maxpool)

    def get_segments(self, checkpoint):
        if isinstance(checkpoint, Iterable):
            segments = [int(x) for x in checkpoint]
        else:
            segments = [int(checkpoint)] * 5
        return segments

    def forward(self, input):
        """
        """
        x = input['image']
        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu(x)
        # c1 = self.maxpool(x)

        # c2 = self.layer1(c1)
        # c3 = self.layer2(c2)
        # c4 = self.layer3(c3)
        # c5 = self.layer4(c4)
        # outs = [c1, c2, c3, c4, c5]
        # features = [outs[i] for i in self.out_layers]
        # return {'features': features, 'strides': self.out_strides}
        outs = []
        for layer_idx in range(0, 5):
            layer = getattr(self, f'layer{layer_idx}', None)
            if layer is not None:  # For C4, layer5_assist is not available
                # use checkpoint for learnable layer
                if self.segments[layer_idx] > 0 and layer_idx not in self.frozen_layers:
                    x = self.checkpoint_fwd(layer, x, self.segments[layer_idx])
                else:
                    x = layer(x)
                outs.append(x)

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.out_strides}


    def _make_layer(self,
                    block,
                    inplanes,
                    outplanes,
                    blocks,
                    stride=1,
                    cardinality=32,
                    base_width=4,
                    deformable=None,
                    normalize={'type': 'solo_bn'}):

        block_deform = [False] * blocks
        if deformable == 'last':
            block_deform[-1] = True
        elif deformable == 'all':
            block_deform = [True] * blocks
        elif isinstance(deformable, int):
            block_deform = [False] * (blocks - deformable) + [True] * deformable

        layers = []
        layers.append(block(inplanes, outplanes, stride, cardinality, base_width, 4,
                            deformable=block_deform[0], normalize=normalize))
        for i in range(1, blocks):
            layers.append(block(outplanes, outplanes, 1, cardinality, base_width, 4,
                                deformable=block_deform[i], normalize=normalize))
        return nn.Sequential(*layers)

    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input.detach_()
            input.requires_grad = True
        return fully_checkpoint_sequential(layer, segments, input)

    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def freeze_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.norm1, self.relu, self.maxpool), self.layer1, self.layer2, self.layer3,
            self.layer4
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


def resnext_101_32x4d(**kwargs):
    """Constructs a ResNeXt-101-32x4d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 32, 4, **kwargs)
    return model


def resnext_101_32x8d(**kwargs):
    """Constructs a ResNeXt-101-32x8d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 32, 8, **kwargs)
    return model


def resnext_101_64x4d(**kwargs):
    """Constructs a ResNeXt-101-64x4d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 64, 4, **kwargs)
    return model


def resnext_101_64x8d(**kwargs):
    """Constructs a ResNeXt-101-64x8d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 64, 8, **kwargs)
    return model


def resnext_152_32x4d(**kwargs):
    """Constructs a ResNeXt-152-32x4d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], 32, 4, **kwargs)
    return model


def resnext_152_32x8d(**kwargs):
    """Constructs a ResNeXt-152-32x8d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], 32, 8, **kwargs)
    return model


def resnext_152_64x4d(**kwargs):
    """Constructs a ResNeXt-152-64x4d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], 64, 4, **kwargs)
    return model


def resnext_152_64x8d(**kwargs):
    """Constructs a ResNeXt-151-64x8d model"""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], 64, 8, **kwargs)
    return model
