import torch
import torch.nn as nn

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class SepConvOperation(nn.Module):
    
    def __init__(self, cfg):
        super(SepConvOperation, self).__init__()
        inplane = cfg['inplane']
        outplane = cfg['outplane']
        kernel_size = cfg['kernel_size']
        stride = cfg['stride']
        padding = cfg['padding']
        affine = cfg.get('affine', True)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(inplane, inplane, kernel_size=kernel_size, stride=stride, padding=padding, groups=inplane, bias=False),
            nn.Conv2d(inplane, inplane, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(inplane, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplane, inplane, kernel_size=kernel_size, stride=1, padding=padding, groups=inplane, bias=False),
            nn.Conv2d(inplane, outplane, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(outplane, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class IdentityOperation(nn.Module):

    def __init__(self, cfg):
        super(IdentityOperation, self).__init__()

    def forward(self, x):
        return x

class ZeroOperation(nn.Module):

    def __init__(self, cfg):
        super(ZeroOperation, self).__init__()
        self.stride = cfg['stride']

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduceOperation(nn.Module):

    def __init__(self, cfg):
        super(FactorizedReduceOperation, self).__init__()
        inplane = cfg['inplane']
        outplane = cfg['outplane']
        affine = cfg.get('affine', True)

        assert outplane % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(inplane, outplane // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(inplane, outplane // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(outplane, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out
