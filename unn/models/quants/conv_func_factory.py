from .irconv2d import IRConv2d

class ConvFuncFactory:

    @classmethod
    def create(cls, name, conv_cfg, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binaryfunction):
        if name == 'irconv2d':
            conv = IRConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binaryfunction, conv_cfg)
        else:
            raise ValueError('Unrecognized Conv Method: ' + name)
        return conv
