import math

import torch
import torch.nn as nn

from ...utils.bn_helper import setup_bn, rollback_bn, FREEZE

__all__ = ["shufflenetv2"]


def conv3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv1x1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def dwconv3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False, groups=inp),
        nn.BatchNorm2d(oup)
    )


def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x


def channel_split(x):
    x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
    return x1, x2


class ShuffleUnitV2S1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShuffleUnitV2S1, self).__init__()

        assert(in_channels == out_channels)
        inp = in_channels // 2
        self.xx = nn.Sequential(
            conv1x1(inp, inp, 1),
            dwconv3x3(inp, inp, 1),
            conv1x1(inp, inp, 1)
        )

    def forward(self, x):
        out1, out2 = channel_split(x)
        out2 = self.xx(out2)
        out = torch.cat((out1, out2), 1)
        out = channel_shuffle(out, groups=2)

        return out


class ShuffleUnitV2S2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShuffleUnitV2S2, self).__init__()

        inp = in_channels
        oup = out_channels
        oup = oup // 2
        self.xx = nn.Sequential(
            conv1x1(inp, inp, 1),
            dwconv3x3(inp, inp, 2),
            conv1x1(inp, oup, 1)
        )
        self.oo = nn.Sequential(
            dwconv3x3(inp, inp, 2),
            conv1x1(inp, oup, 1)
        )

    def forward(self, x):
        out1 = self.oo(x)
        out2 = self.xx(x)
        out = torch.cat((out1, out2), 1)
        out = channel_shuffle(out, groups=2)

        return out


class ShuffleNet(nn.Module):
    def __init__(self, out_strides,
                 groups=3,
                 in_channels=3,
                 bn={FREEZE: True},
                 width_mult=1.0):

        # setup bn before building model
        setup_bn(bn)

        super(ShuffleNet, self).__init__()
        self.out_strides = out_strides

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.width_mult = width_mult

        if width_mult == 0.35:
            self.stage_out_channels = [-1, 16, 32, 64, 128, 512]
        elif width_mult == 0.25:
            self.stage_out_channels = [-1, 12, 24, 48, 96, 512]
        elif width_mult == 0.1:
            self.stage_out_channels = [-1, 6, 12, 24, 48, 256]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError("{} width_mult is not supported".format(width_mult))

        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)

        self.out_planes = [-1, self.stage_out_channels[1]] + self.stage_out_channels[1:-1]
        self.out_planes = [self.out_planes[int(math.log(s, 2))] for s in self.out_strides]
        self.out_layer_idx = [int(math.log(s, 2)) for s in self.out_strides]
        # rollback bn after model builded
        rollback_bn()

    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def _make_stage(self, stage):
        modules = []
        first_module = ShuffleUnitV2S2(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage]
        )
        modules.append(first_module)
        for i in range(self.stage_repeats[stage - 2]):
            module = ShuffleUnitV2S1(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage]
            )
            modules.append(module)
        return nn.Sequential(*modules)

    def forward(self, input):
        x = input['image']
        c1 = self.conv1(x)
        c2 = self.maxpool(c1)

        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        output = [x, c1, c2, c3, c4, c5]
        output = [output[idx] for idx in self.out_layer_idx]

        return {'features': output, 'strides': self.get_outstrides()}


def shufflenetv2(**kwargs):
    model = ShuffleNet(**kwargs)
    return model
