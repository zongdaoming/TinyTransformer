from torch.autograd import Function
import torch

class STEBinaryQuantize(Function):

    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input
