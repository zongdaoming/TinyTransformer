import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..initializer import initialize_from_cfg
from ...extensions import DeformableConv
from ...utils.bn_helper import setup_bn, rollback_bn, FREEZE

import logging
logger = logging.getLogger('global')

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        if dilation == 1:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeformBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=2, downsample=None):
        super(DeformBasicBlock, self).__init__()
        assert stride == 1, 'stride=1 required, but found:{}'.format(stride)
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2

        self.deform_offset = nn.Conv2d(
            inplanes, 18, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv1 = DeformableConv(
            inplanes, planes, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        offset = self.deform_offset(x)

        out = self.conv1(x, offset)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if dilation == 1:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False)
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeformBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=2, downsample=None):
        super(DeformBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2

        if dilation == 2:
            self.deform_offset = nn.Conv2d(
                planes, 18, kernel_size=3, stride=1, padding=padding, dilation=2)
            self.conv2 = DeformableConv(
                planes, planes, kernel_size=3, stride=1, padding=padding, dilation=2)
        else:
            self.deform_offset = nn.Conv2d(
                planes, 18, kernel_size=3, stride=stride, padding=padding, dilation=1)
            self.conv2 = DeformableConv(
                planes, planes, kernel_size=3, stride=stride, padding=padding, dilation=1)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.deform_offset(out)

        out = self.conv2(out, offset)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def make_layer4(inplanes, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, dilation, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, stride=1, dilation=dilation))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 out_layers,
                 out_strides,
                 frozen_layers=[],
                 layer_deform=None,
                 layer4=None,
                 deformable=None,
                 bn={FREEZE: True},
                 initializer=None):
        """
        layer0 <-> Conv1, ..., layer4 <-> Conv5

        Arguments:
            frozen_layers: index of frozen layers, [0,4]
            out_layers: index of output layers, [0,4]
            out_strides: stride of outputs
            layer_deform: list, dcn setting for each layer
            layer4: config of layer4, stride, block, dilation. (Deprecated soon)
            deformable: use dcn or not. (Deprecated soon)
            bn: config of BatchNorm
            initializer: config of initilizaiton
        """

        def check_range(x):
            assert min(x) >= 0 and max(x) <= 4, x

        check_range(frozen_layers)
        check_range(out_layers)
        assert len(out_layers) == len(out_strides)

        # setup bn before building model
        setup_bn(bn)

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        layer_out_planes = [64] + [i * block.expansion for i in [64, 128, 256, 512]]
        self.out_planes = [layer_out_planes[i] for i in out_layers]

        if deformable is not None or layer4 is not None:
            layer_deform, layer4_stride = self.parse_deprecated_style(deformable, layer4)
        else:
            if layer_deform is None or len(layer_deform) == 0:
                layer_deform = [False] * 5
            if 4 in self.out_layers:
                layer4_stride = out_strides[-1] // 16

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, deformable=layer_deform[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, deformable=layer_deform[3])
        if 4 in self.out_layers:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=layer4_stride,
                                           dilation=2 // layer4_stride,
                                           deformable=layer_deform[4])
        else:
            self.layer4 = None

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params
        # to optimizer
        self.freeze_layer()

        # rollback bn after model builded
        rollback_bn()

    def parse_deprecated_style(self, deformable, layer4):
        logger.warning("Argument `deformable` and `layer4` will be deprecated,"
                       "pls use `layer_deform` instead, layer4 setting can be infered from other arguments")
        layer_deform = [False, False, False, False, False]
        if deformable:
            layer_deform = [False, False, 'last', 'last', 'all']
        if layer4 is not None:
            layer4_stride = layer4['stride']
            layer4_block = layer4['block']
            if 'deform' in layer4_block.lower():  # DeformBasicBlock, DeformBlock
                layer_deform[-1] = 'all'
        else:
            layer4_stride = 2
        return layer_deform, layer4_stride

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, deformable=False):
        if deformable in ['False', False, None, 'None', 'none', 0]:
            deformable = None
        deform_block = None
        if deformable is not None:
            if block is BasicBlock:
                deform_block = DeformBasicBlock
            elif block is Bottleneck:
                deform_block = DeformBlock
            else:
                raise ValueError(f'{block.__name__} has no deformable version')
        block_types = [block] * blocks

        if deformable == 'last':
            block_types[-1] = deform_block
        elif deformable == 'all':
            block_types = [deform_block] * blocks
        elif isinstance(deformable, int):
            block_types = [block] * (blocks - deformable) + [deform_block] * deformable

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block_types[0](self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block_types[i](self.inplanes, planes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def forward(self, input):
        x = input['image']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)  # layer1 usually called conv2
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4) if self.layer4 else None

        outs = [c1, c2, c3, c4, c5]
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool),
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

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


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
