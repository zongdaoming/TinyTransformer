"""
  These functions are mainly used in train_val.py
"""

import copy
import json
import logging
import random
import re

import torch
import numpy as np

from .grad_clipper import GradClipper

logger = logging.getLogger('global')


def build_cls_instance(module, cfg):
    """Build instance for given cls"""
    cls = getattr(module, cfg['type'])
    return cls(**cfg['kwargs'])


def build_optimizer(cfg_optim, model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    cfg_optim['kwargs']['params'] = trainable_params
    optimizer = build_cls_instance(torch.optim, cfg_optim)
    logger.info('build optimizer done')
    return optimizer


def build_grad_clipper(cfg_clipper):
    if cfg_clipper is None:
        return None
    clipper = GradClipper(**cfg_clipper)
    logger.info('build gradcliper done')
    return clipper


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def add_table(board, tag, record, step):
    """Add formated table for tensorboard display"""
    keys, vals = zip(*record.items())
    head = '| ' + ' | '.join(keys) + ' |'
    align = '| ' + ' | '.join([':-:'] * len(keys)) + ' |'
    line = '| ' + ' | '.join([f'{_:.3f}' for _ in vals]) + ' |'
    board.add_text(tag, head + '\n' + align + '\n' + line, step)


def format_cfg(cfg):
    """Format experiment config for friendly display"""

    def list2str(cfg):
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = list2str(value)
            elif isinstance(value, list):
                if len(value) == 0 or isinstance(value[0], (int, float)):
                    cfg[key] = str(value)
                else:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            value[i] = list2str(item)
                    cfg[key] = value
        return cfg

    cfg = list2str(copy.deepcopy(cfg))
    json_str = json.dumps(cfg, indent=2, ensure_ascii=False).split(r"\n")
    json_str = [re.sub(r"(\"|,$|\{|\}|\[$)", "", line) for line in json_str if line.strip() not in "{}[]"]
    cfg_str = r"\n".join([line.rstrip() for line in json_str if line.strip()])
    return cfg_str


def to_device(input, device="cuda"):
    """Transfer data between devidces"""

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device)
        elif isinstance(x, list) and torch.is_tensor(x[0]):
            return [_.to(device) for _ in x]
        return x
    return {k: transfer(v) for k, v in input.items()}


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    res = pattern.match(num)
    if res:
        return True
    return False


def try_decode(val):
    """int, float, or str"""
    if val.isdigit():
        return int(val)
    if is_number(val):
        return float(val)
    return val


def merge_opts_into_cfg(opts, cfg):
    cfg = copy.deepcopy(cfg)
    if opts is None or len(opts) == 0:
        return cfg

    assert len(opts) % 2 == 0
    keys, values = opts[0::2], opts[1::2]
    for key, val in zip(keys, values):
        logger.info(f'replacing {key}')
        val = try_decode(val)
        cur_cfg = cfg
        key = key.split('.')
        for k in key[:-1]:
            cur_cfg = cur_cfg.setdefault(k, {})
        cur_cfg[key[-1]] = val
    return cfg
