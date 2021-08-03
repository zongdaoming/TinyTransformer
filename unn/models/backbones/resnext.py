import logging

import torch.nn as nn
import torch.nn.functional as F

from ..initializer import initialize_from_cfg
from ...extensions import DeformableConvInOne
from ...utils.bn_helper import setup_bn, rollback_bn, FREEZE

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

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, deformable=False):
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
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = key_conv(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'shortcut_conv',
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        residual = self.shortcut(x)
        return F.relu(residual + bottleneck, inplace=True)


class ResNeXt(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 cardinality,
                 base_width,
                 out_layers,
                 out_strides,
                 bn={FREEZE: True},
                 frozen_layers=None,
                 deformable=None,
                 layer_deform=None,
                 initializer=None):

        # setup bn before building model
        setup_bn(bn)

        super(ResNeXt, self).__init__()

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

        if layer_deform is None:
            logger.warning("Argument `deformable` will be deprecated" "pls use layer_deform instead")
            if deformable:
                layer_deform = [None, None, 'last', 'last', 'last']
            else:
                layer_deform = [None] * 5
        assert len(layer_deform) == 5, layer_deform
        assert not layer_deform[0] and not layer_deform[1]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 should not use deformable
        self.layer1 = self._make_layer(block, 64, 256, layers[0], 1, cardinality, base_width)
        self.layer2 = self._make_layer(
            block, 256, 512, layers[1], 2, cardinality, base_width, deformable=layer_deform[2])
        self.layer3 = self._make_layer(
            block, 512, 1024, layers[2], 2, cardinality, base_width, deformable=layer_deform[3])
        self.layer4 = self._make_layer(
            block, 1024, 2048, layers[3], 2, cardinality, base_width, deformable=layer_deform[4])

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params
        # to optimizer
        self.freeze_layer()

        # rollback bn after model builded
        rollback_bn()

    def forward(self, input):
        x = input['image']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        outs = [c1, c2, c3, c4, c5]
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.out_strides}

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1, cardinality=32, base_width=4, deformable=None):

        block_deform = [False] * blocks
        if deformable == 'last':
            block_deform[-1] = True
        elif deformable == 'all':
            block_deform = [True] * blocks
        elif isinstance(deformable, int):
            block_deform = [False] * (blocks - deformable) + [True] * deformable

        layers = []
        layers.append(block(inplanes, outplanes, stride, cardinality, base_width, 4, deformable=block_deform[0]))
        for i in range(1, blocks):
            layers.append(block(outplanes, outplanes, 1, cardinality, base_width, 4, deformable=block_deform[i]))
        return nn.Sequential(*layers)

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
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool), self.layer1, self.layer2, self.layer3,
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
