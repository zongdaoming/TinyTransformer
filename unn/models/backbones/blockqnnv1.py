from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from ..initializer import initialize_from_cfg
from ...extensions import DeformableConvInOne
from ...utils.bn_helper import setup_bn, rollback_bn, FREEZE

__all__ = ['bqnnv1_large']


class Block(nn.Module):
    def __init__(self, in_channel, channel, deformable=None):
        """Deformable: indices of op need to use deformalbe conv"""
        super(Block, self).__init__()
        op_conv2d = [nn.Conv2d] * 10
        if deformable:
            for i in deformable:
                op_conv2d[i] = DeformableConvInOne

        self.op1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
            nn.ReLU(),
            op_conv2d[1](
                channel, channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )
        self.op4 = nn.Sequential(
            nn.ReLU(),
            op_conv2d[4](
                in_channel + channel,
                in_channel + channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=in_channel + channel,
                bias=False),
            nn.Conv2d(in_channel + channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )
        self.op5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channel + channel * 2,
                in_channel + channel * 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=in_channel + channel * 2,
                bias=False),
            nn.Conv2d(in_channel + channel * 2, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )
        self.op6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1), groups=in_channel, bias=False),
            nn.Conv2d(in_channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )
        self.op7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )
        self.op8 = nn.Sequential(
            nn.ReLU(),
            op_conv2d[8](
                in_channel + channel * 2,
                in_channel + channel * 2,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
                groups=in_channel + channel * 2,
                bias=False),
            nn.Conv2d(in_channel + channel * 2, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )

        self.op9 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
            nn.ReLU(),
            op_conv2d[9](
                channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )
        self.op10 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1), groups=in_channel, bias=False),
            nn.Conv2d(in_channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel, eps=0.001, momentum=0.0003, affine=True),
        )

    def forward(self, x):
        x0 = x
        x1 = self.op1(x0)
        x2 = torch.cat([x0, x1], 1)
        x3 = torch.cat([x1, x2], 1)
        x4 = self.op4(x2)
        x5 = self.op5(x3)
        x6 = self.op6(x0)
        x7 = self.op7(x6)
        x8 = self.op8(x3)
        x9 = self.op9(x0)
        x10 = self.op10(x0)
        x11 = x6
        x12 = x5 + x8
        x13 = x10
        x = torch.cat([x4, x7, x9, x11, x12, x13], 1)
        return x


class DqnnaNetLargeV1(nn.Module):
    def __init__(self,
                 out_layers,
                 out_strides,
                 bn={FREEZE: True},
                 frozen_layers=None,
                 deformable=None,
                 initializer=None):
        """
        Args:
          out_layers: indices of output layers, {0,1,2,3,4}
          out_strides: strides of output features
          deformable: indices of deformable op in Block
          initiailizer: initializer method
        """

        super(DqnnaNetLargeV1, self).__init__()

        # setup bn before building model
        setup_bn(bn)

        channels = [80, 192, 384, 640]
        Hin = 3

        if frozen_layers is not None and len(frozen_layers) > 0:
            assert min(frozen_layers) >= 0, frozen_layers
            assert max(frozen_layers) <= 4, frozen_layers
        assert min(out_layers) >= 0, out_layers
        assert max(out_layers) <= 4, out_layers
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.outplanes = [channels[i - 1] * 6 for i in self.out_layers]

        self.conv1 = nn.Conv2d(Hin, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # frozen layers should not use deformable conv
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(channels[0], channels[0]),
            Block(channels[0] * 6, channels[0]),
            Block(channels[0] * 6, channels[0]),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(channels[0] * 6, channels[1]),
            Block(channels[1] * 6, channels[1]),
            Block(channels[1] * 6, channels[1], deformable),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(channels[1] * 6, channels[2]),
            Block(channels[2] * 6, channels[2]),
            Block(channels[2] * 6, channels[2]),
            Block(channels[2] * 6, channels[2], deformable),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(channels[2] * 6, channels[3]),
            Block(channels[3] * 6, channels[3]),
            Block(channels[3] * 6, channels[3], deformable),
        )
        if initializer is not None:
            initialize_from_cfg(initializer)
        self.freeze_layer()

        # rollback bn after model builded
        rollback_bn()

    def forward(self, input):
        x = input['image']
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c1 = self.relu(self.bn2(self.conv2(x)))

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        outs = [c1, c2, c3, c4, c5]
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.out_strides}

    def get_outplanes(self):
        return self.outplanes

    def freeze_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.conv2, self.bn2, self.relu, self.maxpool), self.layer1,
            self.layer2, self.layer3, self.layer4
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


def bqnnv1_large(pretrain=False, **kwargs):
    return DqnnaNetLargeV1(**kwargs)
