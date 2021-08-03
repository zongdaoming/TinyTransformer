from unn.models.nas.operations.simple_operation import *


class OperationFactory:

    @classmethod
    def create(cls, name, cfg):
        if name == 'none':
            op = ZeroOperation(cfg)
        elif name == 'avg_pool_3x3':
            op = nn.AvgPool2d(3, stride=cfg['stride'], padding=1)
        elif name == 'max_pool_3x3':
            op = nn.MaxPool2d(3, stride=cfg['stride'], padding=1)
        elif name == 'skip_connect':
            if cfg['stride'] == 1:
                op = IdentityOperation(cfg)
            else:
                op = FactorizedReduceOperation(cfg)
        elif 'sep_conv' in name:
            cfg['outplane'] = cfg['inplane']
            if name == 'sep_conv_3x3':
                cfg['kernel_size'] = 3
                cfg['padding'] = 1
                op = SepConvOperation(cfg)
            elif name == 'sep_conv_5x5':
                cfg['kernel_size'] = 5
                cfg['padding'] = 2
                op = SepConvOperation(cfg)
            elif name == 'sep_conv_7x7':
                cfg['kernel_size'] = 7
                cfg['padding'] = 3
                op = SepConvOperation(cfg)
            else:
                raise ValueError('Unrecognized Operation: ' + name)
        
        else:
            raise ValueError('Unrecognized Operation: ' + name)
        return op
