import logging
import os
import shutil
import torch

logger = logging.getLogger('global')


class Saver:
    def __init__(self, save_dir, model, cfg_path=None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.save_meta(model)
        if cfg_path:
            cfg_name = os.path.basename(cfg_path)
            dst_path = os.path.join(self.save_dir, cfg_name)
            shutil.copy(cfg_path, dst_path)

    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters share common prefix 'module.'"""
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def get_model_from_ckpt(self, ckpt_path):
        """Get model state_dict from checkpoint"""
        assert os.path.exists(ckpt_path), 'No such file'
        logger.info('load checkpoint from {}'.format(ckpt_path))
        device = torch.cuda.current_device()
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
        if 'model' in ckpt_dict:
            state_dict = ckpt_dict['model']
        elif 'state_dict' in ckpt_dict:
            state_dict = ckpt_dict['state_dict']
        else:
            state_dict = ckpt_dict
        state_dict = self.remove_prefix(state_dict, 'module.')
        return state_dict

    def restore_optimizer(self, optimizer, ckpt_path):
        """Get optimizer from checkpoint"""
        assert os.path.exists(ckpt_path), 'No such file'
        logger.info('restore optimizer from {}'.format(ckpt_path))
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
        epoch = ckpt.get('epoch', 0)
        optimizer.load_state_dict(ckpt['optimizer'])
        return optimizer, epoch

    def save(self, epoch, **kwargs):
        """Save model checkpoint for one epoch"""
        os.makedirs(self.save_dir, exist_ok=True)
        ckpt_path = os.path.join(self.save_dir, 'ckpt_e{}.pth'.format(epoch))
        kwargs['epoch'] = epoch
        torch.save(kwargs, ckpt_path)
        return ckpt_path

    def save_meta(self, model):
        """Save model structure"""
        os.makedirs(self.save_dir, exist_ok=True)
        meta_path = os.path.join(self.save_dir, 'ckpt_meta.txt')
        with open(meta_path, 'w') as fid:
            fid.write(str(model))
