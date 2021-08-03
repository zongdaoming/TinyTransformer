import copy
import importlib
import logging

import torch.nn as nn

logger = logging.getLogger('global')


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()
        for cfg_subnet in cfg:
            mname = cfg_subnet['name']
            mtype = cfg_subnet['type']
            kwargs = cfg_subnet['kwargs']
            if cfg_subnet.get('prev', None) is not None:
                prev_module = getattr(self, cfg_subnet['prev'])
                kwargs['inplanes'] = prev_module.get_outplanes()
            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def forward(self, input):
        """
        Note:
            Input should not be updated inplace !!!
            In Mimic task, input may be fed into teacher & student networks respectivly,
            inplace update may cause the input dict only keep the last forward results, which is unexpected.
        """
        input = copy.copy(input)
        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        return input

    def load(self, other_state_dict, strict=False):
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
