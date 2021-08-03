import math
import logging

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

from .._C import deform_conv_v1

logger = logging.getLogger('global')


class DeformableConv(Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 num_deformable_groups=1):
        super(DeformableConv, self).__init__()

        # logger.warning('warning! DeformableConv will be deprecated in the near future, '
        #                'plase use DeformConv2d instead, which is an unified module with '
        #                'the same interface with torch.nn.Conv2d')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_deformable_groups = num_deformable_groups

        assert in_channels % groups == 0
        assert out_channels % groups == 0
        assert (in_channels // groups % num_deformable_groups == 0)
        assert (out_channels // groups % num_deformable_groups == 0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return conv_offset2d(
            input, offset, self.weight, self.stride, self.padding,
            self.dilation, self.groups, self.num_deformable_groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_deformable_groups={num_deformable_groups}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv_offset2d(input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deform_groups=1):
    if input is not None and input.dim() != 4:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = DeformableConvFunction(
        _pair(stride), _pair(padding), _pair(dilation), groups, deform_groups)
    return f(input, offset, weight)


class DeformableConvFunction(Function):

    def __init__(self, stride, padding, dilation, groups=1, deformable_groups=1):
        super(DeformableConvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, weight):

        self.save_for_backward(input, offset, weight)

        output = input.new(*self._output_size(input, weight))

        self.bufs_ = [
            input.new(input.size(1) * weight.size(3) * weight.size(2), output.size(2), output.size(3)).zero_(),
            input.new(output.size(2), output.size(3)).fill_(1)
        ]  # columns, ones

        forward_fn = deform_conv_v1.forward_cuda
        if not input.is_cuda:
            logger.warning(
                '---CPU version of DEFORMABLE CONV V1 is a dummpy function, which is used to support tocaffe')
            forward_fn = deform_conv_v1.forward_cpu

        forward_fn(
            input, weight, offset, output,
            self.bufs_[0], self.bufs_[1],
            weight.size(2), weight.size(3),
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups, self.deformable_groups)
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = None
        grad_offset = None
        grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            grad_output = grad_output.contiguous()
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()

                deform_conv_v1.backward_input_cuda(
                    input, offset, grad_output, grad_input, grad_offset, weight, self.bufs_[0],
                    weight.size(2), weight.size(3), self.stride[0], self.stride[1],
                    self.padding[0], self.padding[1], self.dilation[0],
                    self.dilation[1], self.groups, self.deformable_groups)

            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv_v1.backward_parameters_cuda(
                    input, offset, grad_output, grad_weight, self.bufs_[0], self.bufs_[1],
                    weight.size(2), weight.size(3), self.stride[0], self.stride[1],
                    self.padding[0], self.padding[1], self.dilation[0],
                    self.dilation[1], self.groups, self.deformable_groups, 1)
        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )

        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                'x'.join(map(str, output_size))))
        return output_size


class DeformConv2d(Module):
    """Unified Deformable Convolution V1 sharing the same interface with nn.Conv2d"""

    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deform_groups=1):
        super(DeformConv2d, self).__init__()

        ks = _pair(kernel_size)

        self.offset_conv = nn.Conv2d(
            inplanes,
            2 * ks[0] * ks[1],
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias)

        self.deform_conv = DeformableConv(
            inplanes,
            outplanes,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            num_deformable_groups=deform_groups)

        self.reset_parameters()

    def reset_parameters(self):
        self.offset_conv.weight.data.normal_(std=0.01)
        if self.offset_conv.bias is not None:
            self.offset_conv.bias.data.zero_()

    def forward(self, input):
        return self.deform_conv(input, self.offset_conv(input))


class DeformableConvInOne(Module):
    """
    Unified Deformable Convolution V1 sharing the same interface with nn.Conv2d,
    also can reuse pretrained params of nn.Conv2d
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 num_deformable_groups=1):
        super(DeformableConvInOne, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_deformable_groups = num_deformable_groups

        assert in_channels % groups == 0
        assert out_channels % groups == 0
        assert (in_channels // groups % num_deformable_groups == 0)
        assert (out_channels // groups % num_deformable_groups == 0)
        assert bias is False

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        ks = _pair(kernel_size)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * ks[0] * ks[1],
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.offset_conv.weight.data.normal_(std=0.01)
        if self.offset_conv.bias is not None:
            self.offset_conv.bias.data.zero_()

    def forward(self, input):
        offset = self.offset_conv(input)
        f = DeformableConvFunction(
            self.stride, self.padding, self.dilation, self.groups, self.num_deformable_groups)
        return f(input, offset, self.weight)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_deformable_groups={num_deformable_groups}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
