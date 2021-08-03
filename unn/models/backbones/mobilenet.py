import math
import torch.nn as nn

from ...utils.bn_helper import setup_bn, rollback_bn, FREEZE

__all__ = ["mobilenetv2"]


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet(nn.Module):

    def __init__(self, out_strides,
                 width_mult=1.0,
                 bn={FREEZE: True}):

        # setup bn before building model
        setup_bn(bn)

        super(MobileNet, self).__init__()
        self.out_strides = out_strides

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self.features = [conv_bn(3, input_channel, 2)]
        layer_index = [-1]  # last layer before downsample
        output_channels = [-1]

        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            if s == 2:
                layer_index.append(len(self.features) - 1)
                output_channels.append(input_channel)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        layer_index.append(len(self.features) - 1)
        output_channels.append(self.last_channel)

        self.features = nn.Sequential(*self.features)

        assert(len(output_channels) == 6)
        assert(len(layer_index) == 6)
        self.out_planes = [output_channels[int(math.log(s, 2))] for s in self.out_strides]
        self.out_layer_index = [layer_index[int(math.log(s, 2))] for s in self.out_strides]

        # rollback bn after model builded
        rollback_bn()

    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def forward(self, input):
        x = input['image']
        output = []
        k = 0
        for m_idx, module in enumerate(self.features._modules.values()):
            x = module(x)
            if k < len(self.out_layer_index) and m_idx == self.out_layer_index[k]:
                output.append(x)
                k += 1
        assert len(output) == len(self.out_layer_index), '{} vs {}'.format(
            len(output), len(self.out_layer_index))
        return {'features': output, 'strides': self.get_outstrides()}


def mobilenetv2(**kwargs):
    model = MobileNet(**kwargs)
    return model
