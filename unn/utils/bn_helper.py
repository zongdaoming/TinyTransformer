import logging

import numpy as np
import torch
import linklink as link
from .dist_helper import get_rank
from .dist_helper import get_world_size

logger = logging.getLogger('global')

# bn mode
SYNC = 'sync'
SOLO = 'solo'
FREEZE = 'freeze'
# gn mode
GN = 'group_norm'
CAFFE_FREEZE = 'caffe_freeze'


def setup_bn(cfg):
    """
    bn mode: default solo
    sync mode: Sync mean and var across gpus
    freeze mode: freeze mean and var
    solo mode: compute mean and var solely
    """
    assert torch.nn.BatchNorm2d.__name__ == 'BatchNorm2d', 'please call `rollback_bn()` first'
    torch.nn._BatchNorm2d = torch.nn.BatchNorm2d
    if cfg.get(SYNC, None):
        rank = get_rank()
        world_size = get_world_size()
        cfg_bn = cfg.get('sync')
        bn_group_size = cfg_bn['bn_group_size']
        bn_momentum = cfg_bn.get('bn_momentum', 0.1)
        assert world_size % bn_group_size == 0
        bn_group_comm = simple_group_split(world_size, rank, world_size // bn_group_size)
        DeprecatedGroupSyncBatchNorm.set_sync(
            bn_momentum=bn_momentum,
            group=bn_group_comm,
            sync_stats=True)
        torch.nn.BatchNorm2d = DeprecatedGroupSyncBatchNorm
        mode = SYNC
    elif cfg.get(FREEZE, False):
        torch.nn.BatchNorm2d = FrozenBatchNorm2d
        mode = FREEZE
    elif cfg.get(SOLO, False):
        mode = SOLO
    elif cfg.get(GN, False):
        cfg_gn = cfg.get(GN)
        DeprecatedGroupNorm.num_group = cfg_gn.get('num_groups', 32)
        torch.nn.BatchNorm2d = DeprecatedGroupNorm
        mode = GN
    elif cfg.get(CAFFE_FREEZE, False):
        torch.nn.BatchNorm2d = CaffeFrozenBatchNorm2d
        mode = CAFFE_FREEZE
    else:
        raise NotImplementedError(f'bn setting {cfg} is not supported')
    logger.info(f'setup BN in {mode} mode')
    return mode


def rollback_bn():
    assert hasattr(torch.nn, '_BatchNorm2d'), 'please call setup_bn before rolling back to original bn'
    torch.nn.BatchNorm2d = torch.nn._BatchNorm2d


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(ranks=rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]


# TODO this class will be deprecated
class DeprecatedGroupSyncBatchNorm(link.nn.SyncBatchNorm2d):
    @staticmethod
    def set_sync(bn_momentum=0.1, group=None, sync_stats=False, var_mode=link.syncbnVarMode_t.L2):
        DeprecatedGroupSyncBatchNorm.bn_momentum = bn_momentum
        DeprecatedGroupSyncBatchNorm.group = group
        DeprecatedGroupSyncBatchNorm.sync_stats = sync_stats
        DeprecatedGroupSyncBatchNorm.var_mode = var_mode

    def __init__(self, *_args, **_kwargs):
        super(DeprecatedGroupSyncBatchNorm, self).__init__(
            *_args,
            **_kwargs,
            momentum=self.bn_momentum,
            group=self.group,
            sync_stats=self.sync_stats,
            var_mode=self.var_mode
        )

    def __repr__(self):
        return ('{name}({num_features},'
                ' eps={eps},'
                ' momentum={momentum},'
                ' affine={affine},'
                ' group={group},'
                ' sync_stats={sync_stats},'
                ' var_mode={var_mode})'.format(
                    name=self.__class__.__name__, **self.__dict__))


# TODO this class will be deprecated
class DeprecatedGroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels):
        super(DeprecatedGroupNorm, self).__init__(self.num_group, num_channels)


class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    def _init__(self, n):
        super(FrozenBatchNorm2d, self).__init__(n)
        self.training = False

    def train(self, mode=False):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class CaffeFrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed.
    This implementation is different from pod.utils.bn_helper.FrozenBatchNorm2d in that
        - 1. weight & bias are also frozen in this version
        - 2. there is no eps when computing running_var.rsqrt()
    """

    def __init__(self, n):
        super(CaffeFrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class GroupSyncBatchNorm(link.nn.SyncBatchNorm2d):

    def __init__(self,
                 num_features,
                 bn_group_size=None,
                 momentum=0.1,
                 sync_stats=True,
                 var_mode=link.syncbnVarMode_t.L2):

        self.group_size = bn_group_size

        super(GroupSyncBatchNorm, self).__init__(
            num_features,
            momentum=momentum,
            group=self._get_group(bn_group_size),
            sync_stats=sync_stats,
            var_mode=var_mode
        )

    @staticmethod
    def _get_group(bn_group_size):
        rank = get_rank()
        world_size = get_world_size()
        if bn_group_size is None:
            bn_group_size = world_size
        assert world_size % bn_group_size == 0
        bn_group_comm = simple_group_split(world_size, rank, world_size // bn_group_size)
        return bn_group_comm

    def __repr__(self):
        return ('{name}({num_features},'
                ' eps={eps},'
                ' momentum={momentum},'
                ' affine={affine},'
                ' group={group},'
                ' group_size={group_size},'
                ' sync_stats={sync_stats},'
                ' var_mode={var_mode})'.format(
                    name=self.__class__.__name__, **self.__dict__))


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        kwargs['num_channels'] = num_channels
        super(GroupNorm, self).__init__(**kwargs)
