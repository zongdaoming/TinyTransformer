from __future__ import division
import logging
import os
import time
import pdb
import nart_tools.pytorch as pytorch
import torch
from unn.models.networks.network_factory import NetworkFactory
from unn.utils.log_helper import init_log
from unn.utils.saver_helper import Saver

import nart_tools
print('nart_tools version: {}'.format(nart_tools.__version__))

init_log('global', logging.INFO)
logger = logging.getLogger('global')


def prepare():
    # warkaround to avoid taking `size()` into compuation graph,
    # which nart_tools is incapable to handle
    torch.Tensor._shpae = torch.Tensor.shape
    @property
    def shape(self):
        return self.detach()._shpae
    torch.Tensor.shape = shape

    # torch.Tensor._float = torch.Tensor.float
    # def float(self):
    #     logger.info(f'dtype:{self.dtype}')
    #     if self.dtype == torch.float16:
    #         return self._float()
    #     return self
    # torch.Tensor.float = float


class Wrapper(torch.nn.Module):
    def __init__(self, detector):
        super(Wrapper, self).__init__()
        self.detector = detector

    def forward(self, input, get_name=False):
        logger.info(f'before detector forward')
        output = self.detector(input)
        logger.info(f'detector output:{output.keys()}')
        blob_names = []
        blob_datas = []
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blobs') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
                logger.info(f'blobs:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        return blob_names if get_name else blob_datas


def tocaffe(cfg, save_prefix, input_size=(128 * 4, 128 * 4)):
    save_dir = os.path.dirname(save_prefix)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    h, w = input_size
    image_shape = (1, 3, h, w)
    input = {
        'image_info': torch.tensor([h, w, 1.0, h, w, 0])[None, :],
        'image': torch.randn(*image_shape)
    }

    model = NetworkFactory.create(cfg['network']['name'], cfg['network']['cfg'])

    ckpt_path = cfg['saver'].get('resume_model', None)
    if ckpt_path:
        saver = Saver(cfg['saver']['save_dir'], model)
        assert os.path.exists(ckpt_path)
        state_dict = saver.get_model_from_ckpt(ckpt_path)
        model_dict = model.state_dict()
        loaded_num = model.load(state_dict, strict=False)
        if loaded_num != len(model_dict.keys()):
            logger.warning(f'checkpoint keys mismatch loaded keys:({len(model_dict.keys())} vs {loaded_num})')
        # assert loaded_num == len(model_dict.keys()), 'keys mismatch'
    else:
        logger.warning('********no model weights provide********')

    time.sleep(10)
    model = Wrapper(model).eval()
    output_names = model(input, get_name=True)

    input_names = ['data']
    with pytorch.convert_mode():
        pytorch.convert(
            model, [image_shape],
            filename=save_prefix,
            input_dict=input,
            input_names=input_names,
            output_names=output_names,
            verbose=True
        )
