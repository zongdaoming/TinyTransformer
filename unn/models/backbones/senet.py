import math

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections.abc import Iterable

from ..initializer import initialize_from_cfg
from ...extensions import DeformableConvInOne
from ..normalize import build_norm_layer, parse_deprecated_bn_style
from ...utils.checkpoint import fully_checkpoint_sequential

__all__ = [
    'SENet',
    'senet154',
    'se_resnet50',
    'se_resnet101',
    'se_resnet152',
    'se_resnext50_32x4d',
    'se_resnext101_32x4d',
    'se_resnext101_64x4d']


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def extra_repr(self):
        return 'output_size={}'.format(self.output_size)

    def forward(self, x):
        data_type = x.dtype
        out = F.adaptive_avg_pool2d(x.float(), self.output_size)
        out = out.to(data_type)
        return out

def Sigmoid_Activate(input):
    return input * F.sigmoid(input)

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """Base class for bottlenecks that implements `forward()` method"""

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        #out = self.relu(out)
        out = Sigmoid_Activate(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = Sigmoid_Activate(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        #out = self.relu(out)
        out = Sigmoid_Activate(out)

        return out


class SEBottleneck(Bottleneck):
    """Bottleneck for SENet154"""

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None,
                 deformable=False,
                 normalize={'type': 'solo_bn'}):
        super(SEBottleneck, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(planes * 2, normalize, 1)
        self.norm2_name, norm2 = build_norm_layer(planes * 4, normalize, 2)
        self.norm3_name, norm3 = build_norm_layer(planes * self.expansion, normalize, 3)

        key_conv = nn.Conv2d
        if deformable:
            key_conv = DeformableConvInOne
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = key_conv(planes * 2, planes * 4, kernel_size=3, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None,
                 deformable=False,
                 normalize={'type': 'solo_bn'}):
        super(SEResNetBottleneck, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(planes, normalize, 1)
        self.norm2_name, norm2 = build_norm_layer(planes, normalize, 2)
        self.norm3_name, norm3 = build_norm_layer(planes * self.expansion, normalize, 3)

        key_conv = nn.Conv2d
        if deformable:
            key_conv = DeformableConvInOne

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = key_conv(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """ResNeXt bottleneck type C with a Squeeze-and-Excitation module"""

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4, deformable=False,
                 normalize={'type': 'solo_bn'}):
        super(SEResNeXtBottleneck, self).__init__()

        key_conv = nn.Conv2d
        if deformable:
            key_conv = DeformableConvInOne

        width = math.floor(planes * (base_width / 64)) * groups

        self.norm1_name, norm1 = build_norm_layer(width, normalize, 1)
        self.norm2_name, norm2 = build_norm_layer(width, normalize, 2)
        self.norm3_name, norm3 = build_norm_layer(planes * self.expansion, normalize, 3)

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = key_conv(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):
    """
    """
    def __init__(self, block, layers, groups, reduction,
                 out_layers, out_strides,
                 frozen_layers=None,
                 deformable=None,
                 initializer=None,
                 inplanes=128,
                 bn=None,
                 normalize={'type': 'freeze_bn'},
                 checkpoint=False,
                 input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1):
        """
        Arguments:
            - block (``nn.Module``): ``Bottleneck`` class::

                - For SENet154: SEBottleneck
                - For SE-ResNet models: SEResNetBottleneck
                - For SE-ResNeXt models:  SEResNeXtBottleneck

            - layers (:obj:`list` of :obj:`int`): Number of residual blocks for 4 layers of the
              network (layer1...layer4).
            - groups (:obj:`int`): Number of groups for the 3x3 convolution in each::

                bottleneck block.
                - For SENet154: 64
                - For SE-ResNet models: 1
                - For SE-ResNeXt models:  32

            - reduction (:obj:`int`): Reduction ratio for Squeeze-and-Excitation modules::

                - For all models: 16

            - dropout_p (:obj:`float` or None): Drop probability for the Dropout layer::

                If `None` the Dropout layer is not used.
                - For SENet154: 0.2
                - For SE-ResNet models: None
                - For SE-ResNeXt models: None

            - inplanes (:obj:`int`):  Number of input channels for layer1::

                - For SENet154: 128
                - For SE-ResNet models: 64
                - For SE-ResNeXt models: 64

            - input_3x3 (:obj:`bool`): If :obj:`True`, use three 3x3
              convolutions instead of::

                a single 7x7 convolution in layer0.
                - For SENet154: True
                - For SE-ResNet models: False
                - For SE-ResNeXt models: False

            - downsample_kernel_size (:obj:`int`): Kernel size
              for downsampling convolutions in layer2, layer3 and layer4::

                - For SENet154: 3
                - For SE-ResNet models: 1
                - For SE-ResNeXt models: 1

            - downsample_padding (:obj:`int`): Padding for downsampling
              convolutions in layer2, layer3 and layer4::

                - For SENet154: 1
                - For SE-ResNet models: 0
                - For SE-ResNeXt models: 0

            - bn (:obj:`dict`): Deprecated (see normalize). Config of BatchNorm (see Configuration#Normalization).
            - normalize (:obj:`dict`): Config of Normalization Layer (see Configuration#Normalization).
        """
        super(SENet, self).__init__()

        if bn is not None:
            normalize = parse_deprecated_bn_style(bn)

        self.segments = self.get_segments(checkpoint)
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                build_norm_layer(64, normalize, 1),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                build_norm_layer(64, normalize, 2),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                build_norm_layer(inplanes, normalize, 3),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                build_norm_layer(inplanes, normalize, 1),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
            deformable=deformable,
            normalize=normalize
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            deformable=deformable,
            normalize=normalize
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            deformable=deformable,
            normalize=normalize
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            deformable=deformable,
            normalize=normalize
        )

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

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params
        # to optimizer
        self.freeze_layer()

    def get_segments(self, checkpoint):
        if isinstance(checkpoint, Iterable):
            segments = [int(x) for x in checkpoint]
        else:
            segments = [int(checkpoint)] * 5
        return segments

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0,
                    deformable=False,
                    normalize={'type': 'solo_bn'}):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                build_norm_layer(planes * block.expansion, normalize)[1]
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            groups, reduction, stride, downsample,
                            normalize=normalize))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups,
                                reduction, normalize=normalize))
        if deformable:
            layers[-1] = block(self.inplanes, planes, groups,
                               reduction, deformable=deformable, normalize=normalize)

        return nn.Sequential(*layers)

    def get_outplanes(self):
        """
        """
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
        layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input.detach_()
            input.requires_grad = True
        return fully_checkpoint_sequential(layer, segments, input)

    def forward(self, input):
        """

        Arguments:
            - input (:obj:`dict`): output of
              :class:`~pod.datasets.base_dataset.BaseDataset`

        Returns:
            - out (:obj:`dict`):

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        x = input['image']
        outs = []
        for layer_idx in range(0, 5):
            layer = getattr(self, f'layer{layer_idx}', None)
            if layer is not None:  # layer4 is None for C4 backbone
                # Use checkpoint for learnable layer
                if self.segments[layer_idx] > 0 and layer_idx not in self.frozen_layers:
                    x = self.checkpoint_fwd(layer, x, self.segments[layer_idx])
                else:
                    x = layer(x)
                outs.append(x)

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}


def senet154(**kwargs):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16, **kwargs)
    return model


def se_resnet50(**kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, **kwargs)
    return model


def se_resnet101(**kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, **kwargs)
    return model


def se_resnet152(**kwargs):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, **kwargs)
    return model


def se_resnext50_32x4d(**kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, **kwargs)
    return model


def se_resnext101_32x4d(**kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, **kwargs)
    return model


def se_resnext101_64x4d(**kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=64, reduction=16, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, **kwargs)
    return model
