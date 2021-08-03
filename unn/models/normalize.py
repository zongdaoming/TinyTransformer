import torch
from unn.utils.bn_helper import FrozenBatchNorm2d, CaffeFrozenBatchNorm2d, GroupSyncBatchNorm, GroupNorm
from unn.utils.bn_helper import SYNC, FREEZE, SOLO, GN, CAFFE_FREEZE

import logging

logger = logging.getLogger('global')


_norm_cfg = {
    'solo_bn': ('bn', torch.nn.BatchNorm2d),
    'freeze_bn': ('bn', FrozenBatchNorm2d),
    'caffe_freeze_bn': ('bn', CaffeFrozenBatchNorm2d),
    'sync_bn': ('bn', GroupSyncBatchNorm),
    'gn': ('gn', GroupNorm),
}


def build_norm_layer(num_features, cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if layer_type not in _norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = _norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = norm_layer(num_features, **kwargs)
    return name, layer


def parse_deprecated_bn_style(bn):
    logger.warning('Argument bn will be deprecated, please use normalize instead')
    if bn.get(SYNC, False):
        return {
            'type': 'sync_bn',
            'kwargs': bn.get(SYNC) if isinstance(bn.get(SYNC), dict) else {}
        }
    elif bn.get(FREEZE, False):
        return {
            'type': 'freeze_bn',
            'kwargs': bn.get(FREEZE) if isinstance(bn.get(FREEZE), dict) else {}
        }
    elif bn.get(SOLO, False):
        return {
            'type': 'solo_bn',
            'kwargs': bn.get(SOLO) if isinstance(bn.get(SOLO), dict) else {}
        }
    elif bn.get(GN, False):
        return {
            'type': 'gn',
            'kwargs': bn.get(GN) if isinstance(bn.get(GN), dict) else {}
        }
    elif bn.get(CAFFE_FREEZE, False):
        return {
            'type': 'caffe_freeze_bn',
            'kwargs': bn.get(CAFFE_FREEZE) if isinstance(bn.get(CAFFE_FREEZE), dict) else {}
        }
