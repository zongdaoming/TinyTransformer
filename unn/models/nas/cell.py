import torch
import torch.nn as nn
from unn.models.nas.operations.operation_factory import OperationFactory
from unn.models.nas.operations.simple_operation import ReLUConvBN, FactorizedReduce

class MixedOp(nn.Module):

    def __init__(self, channel, stride, cfg):
        super(MixedOp, self).__init__()
        self.channel = channel
        self.stride = stride
        self.create_ops(cfg)

    def create_ops(self, cfg):
        self._ops = nn.ModuleList()
        for item in cfg:
            name = item['name']
            cfg = item.get('cfg', {})
            cfg['stride'] = self.stride
            cfg['inplane'] = self.channel
            cfg['outplane'] = self.channel
            op = OperationFactory.create(name, cfg)
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(self.channel, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, cfg):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, cfg)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
