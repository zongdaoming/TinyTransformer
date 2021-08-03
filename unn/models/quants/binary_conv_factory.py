from .conv_func_factory import ConvFuncFactory
from .binary.binary_quantize_factory import BinaryQuantizeFactory

class BinaryConvFactory:

    @classmethod
    def create(cls, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        conv_name = cls.conv_cfg['name']
        binaryfunction = BinaryQuantizeFactory.create(cls.binary_name)
        conv = ConvFuncFactory.create(conv_name, cls.conv_cfg, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, binaryfunction)
        return conv

    @classmethod
    def init(cls, conv_cfg, binary_name):
        cls.conv_cfg = conv_cfg
        cls.binary_name = binary_name
        
        
