import torch
import torch.nn
import linklink as link

from unn.utils.bn_helper import GroupSyncBatchNorm, FrozenBatchNorm2d
from unn.utils.dist_helper import get_world_size

def op32_wrapper(func):
    """Wrapper forward whose input are all tensors and can only handle float"""

    def forward(self, *args, **kwargs):
        input_types = [v.data.type() for v in list(args) + list(kwargs.values())]
        input_type = input_types[0]
        for t in input_types:
            assert t == input_type, f'{t} vs {input_type}'
        if input_type.lower().find('half') >= 0:
            args_fp32 = [v.float() for v in args]
            kwargs_fp32 = {k: v.float() for k, v in kwargs.items()}
        else:
            args_fp32 = args
            kwargs_fp32 = kwargs
        output = func(self, *args_fp32, **kwargs_fp32)
        if output.data.type() != input_type:
            output = output.type(input_type)
        return output
    return forward

def setup_fp16():
    """Use fp32 for BatchNormal"""
    torch.nn.BatchNorm2d.forward = op32_wrapper(torch.nn.BatchNorm2d.forward)
    GroupSyncBatchNorm.forward = op32_wrapper(GroupSyncBatchNorm.forward)
    FrozenBatchNorm2d.forward = op32_wrapper(FrozenBatchNorm2d.forward)

def copy_grad(params, params_with_grad):
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(torch.zeros_like(param.data))
        if param_w_grad.grad is not None:
            param.grad.data.copy_(param_w_grad.grad.data)

def copy_param(model_params, params_fp32):
    for p_model, p_fp32 in zip(model_params, params_fp32):
        p_model.data.copy_(p_fp32.data)

def params_to_fp32(model):
    params_fp32 = [
        param.clone().type(torch.cuda.FloatTensor).detach()
        for param in model.parameters() if param.requires_grad
    ]
    for param in params_fp32:
        param.requires_grad = True
    return params_fp32


