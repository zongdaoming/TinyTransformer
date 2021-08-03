import copy
import importlib
import logging
import pdb

import torch.nn as nn
import linklink as link

from unn.utils.dist_helper import get_world_size
from unn.utils.dist_helper import reduce_gradients
from unn.models.networks.fp16_helper import copy_grad, copy_param, params_to_fp32

logger = logging.getLogger('global')


class BaseNetwork(nn.Module):

    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.build(cfg)
        self.fp16 = False
        self.cfg = copy.deepcopy(cfg)

    def build(self, cfg):
        self._auto_build(cfg)
    
    def load(self, other_state_dict, strict=False):
        self._auto_load(other_state_dict, strict=False)
    
    def forward(self, input):
        return self._auto_forward(input)

    def to_fp16(self, scale_factor, lr_scheduler):
        self.half()
        self.fp16 = True
        for m in self.modules():
            if m.__class__.__name__.lower().find('batchnorm') >= 0:
                m.float()
        if lr_scheduler is not None:
            self.scale_factor = scale_factor
            self.lr_scheduler = lr_scheduler
            self.params_copy = params_to_fp32(self)
            self.params_model = [p for p in self.parameters() if p.requires_grad]
            optimizer = self.lr_scheduler.optimizer
            optim_state = optimizer.state_dict()
            optimizer.state.clear()
            optimizer.param_groups = []
            optimizer.add_param_group({'params': self.params_copy})
            optimizer.load_state_dict(optim_state)

    def fp16_step(self):
        copy_grad(self.params_copy, self.params_model)
        if get_world_size() > 1:
            for param in self.params_copy:
                if self.scale_factor != 1:
                    param.grad.data /= self.scale_factor
                link.allreduce(param.grad.data)
        self.lr_scheduler.optimizer.step()
        copy_param(self.params_model, self.params_copy)

    def _auto_build(self, cfg):
        for cfg_subnet in cfg:
            mname = cfg_subnet['name']
            mtype = cfg_subnet['type']
            kwargs = cfg_subnet['kwargs']
            if cfg_subnet.get('prev', None) is not None:
                prev_module = getattr(self, cfg_subnet['prev'])
                kwargs['inplanes'] = prev_module.get_outplanes()
            module = self._auto_build_component(mtype, kwargs)
            self.add_module(mname, module)

    def _auto_build_component(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def _auto_forward(self, input):
        """
        Note:
            Input should not be updated inplace !!!
            In Mimic task, input may be fed into teacher & student networks respectivly,
            inplace update may cause the input dict only keep the last forward results, which is unexpected.
        """
        if self.fp16:
            input['image'] = input['image'].half()
        input = copy.copy(input)
        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        return input

    def _auto_load(self, other_state_dict, strict=False):
        """
        1. load resume model or pretained detection model
        2. load pretrained clssification model
        """
        logger.info("try to load the whole resume model or pretrained detection model...")
        model_keys = self.state_dict().keys()
        other_keys = other_state_dict.keys()
        shared_keys, unexpected_keys, missing_keys \
            = self.check_keys(model_keys, other_keys, 'model')
        self.load_state_dict(other_state_dict, strict=strict)

        num_share_keys = len(shared_keys)
        if num_share_keys == 0:
            logger.info(
                'failed to load the whole detection model directly,'
                'try to load each part seperately...'
            )
            for mname, module in self.named_children():
                module.load_state_dict(other_state_dict, strict=strict)
                module_keys = module.state_dict().keys()
                other_keys = other_state_dict.keys()

                # check and display info module by module
                shared_keys, unexpected_keys, missing_keys, \
                    = self.check_keys(module_keys, other_keys, mname)
                self.display_info(mname, shared_keys, unexpected_keys, missing_keys)
                num_share_keys += len(shared_keys)
        else:
            self.display_info("model", shared_keys, unexpected_keys, missing_keys)
        return num_share_keys

    def check_keys(self, own_keys, other_keys, own_name):
        own_keys = set(own_keys)
        other_keys = set(other_keys)
        shared_keys = own_keys & other_keys
        unexpected_keys = other_keys - own_keys
        missing_keys = own_keys - other_keys
        return shared_keys, unexpected_keys, missing_keys

    def display_info(self, mname, shared_keys, unexpected_keys, missing_keys):
        info = "load {}:{} shared keys, {} unexpected keys, {} missing keys.".format(
            mname, len(shared_keys), len(unexpected_keys), len(missing_keys))

        if len(missing_keys) > 0:
            info += "\nmissing keys are as follows:\n    {}".format("\n    ".join(missing_keys))
        logger.info(info)

    def _loss(self, input):
        output = self(input)
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        world_size = get_world_size()
        loss = sum(losses) / world_size
        if self.fp16:
            loss = loss * self.scale_factor
        return loss

    def prepare_gradient(self, output, world_size):
        self.zero_grad()
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        loss = sum(losses) / world_size
        if self.fp16:
            loss = loss * self.scale_factor
        loss.backward()
        if self.fp16:
            loss = loss / self.scale_factor
        output['All.loss'] = loss * world_size

    def update_parameter(self, grad_clipper, distributed, lr_scheduler):
        if grad_clipper:
            grad_clipper.clip_grad(self.parameters())
        if self.fp16:
            self.fp16_step()
        else:
            if distributed:
                reduce_gradients(self, True)
            lr_scheduler.optimizer.step()
