from .ste_binary_quantize import STEBinaryQuantize

class BinaryQuantizeFactory:

    @classmethod
    def create(cls, name):
        if name == 'ste':
            binaryfunction = STEBinaryQuantize
        else:
            raise ValueError('Unrecognized Binary Quantize: ' + name)
        return binaryfunction
