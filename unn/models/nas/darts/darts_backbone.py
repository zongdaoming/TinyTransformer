import torch
import logging
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from unn.models.nas.cell import Cell
__all__ = ['DartsBackbone']

logger = logging.getLogger('global')

class DartsBackbone(nn.Module):

    def __init__(self, C, num_classes, layers, cfg, steps=4, multiplier=4, stem_multiplier=3):
        super(DartsBackbone, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.op_cfg = cfg['operations']

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
 
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.op_cfg)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.outplanes = [C_prev]
        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input['image'])
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        return {'features': [s1], 'strides': [16]}

    def get_outplanes(self):
        return self.outplanes

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(self.op_cfg)

        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
