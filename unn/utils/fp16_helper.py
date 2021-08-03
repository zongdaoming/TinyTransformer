import logging

import torch
import torch.nn
import linklink as link

from .bn_helper import GroupSyncBatchNorm, FrozenBatchNorm2d
from .dist_helper import get_world_size

logger = logging.getLogger('global')


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


def to_fp16(model):
    model_fp16 = model.half()
    for m in model_fp16.modules():
        if m.__class__.__name__.lower().find('batchnorm') >= 0:
            m.float()
    return model_fp16


class Fp16Helper(object):
    def __init__(self, model, scale_factor=None, lr_scheduler=None):
        self.model = to_fp16(model)
        if lr_scheduler is not None:
            self.scale_factor = scale_factor
            self.lr_scheduler = lr_scheduler
            self.params_copy = params_to_fp32(model)
            self.params_model = [p for p in model.parameters() if p.requires_grad]
            optimizer = self.lr_scheduler.optimizer
            optim_state = optimizer.state_dict()
            optimizer.state.clear()
            optimizer.param_groups = []
            optimizer.add_param_group({'params': self.params_copy})
            optimizer.load_state_dict(optim_state)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def __call__(self, input):
        input['image'] = input['image'].half()
        return self.model(input)

    def step(self):
        copy_grad(self.params_copy, self.params_model)
        if get_world_size() > 1:
            for param in self.params_copy:
                if self.scale_factor != 1:
                    param.grad.data /= self.scale_factor
                link.allreduce(param.grad.data)
        self.lr_scheduler.optimizer.step()
        copy_param(self.params_model, self.params_copy)
