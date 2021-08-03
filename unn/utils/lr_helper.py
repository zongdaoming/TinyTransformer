import math
import logging

import torch
from torch.optim import Optimizer
from .misc_helper import build_cls_instance

logger = logging.getLogger('global')


class _IterLRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))

        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_iter + 1)
        self.last_iter = last_iter

    def get_lr(self):
        raise NotImplementedError

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class IterLinearLR(_IterLRScheduler):
    """
    Set the learning rate of each parameter group to the initial lr decayed
    by gamma every iteration. When last_iter=-1, sets initial lr as lr.

    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_iter (int): The index of last iter. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_iter=-1):
        self.gamma = gamma
        super(IterLinearLR, self).__init__(optimizer, last_iter)

    def get_lr(self):
        return [base_lr + self.gamma * self.last_iter for base_lr in self.base_lrs]


class IterExponentialLR(_IterLRScheduler):
    """
    Set the learning rate of each parameter group to the initial lr decayed
    by gamma every iteration. When last_iter=-1, sets initial lr as lr.

    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_iter (int): The index of last iter. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_iter=-1):
        self.gamma = gamma
        super(IterExponentialLR, self).__init__(optimizer, last_iter)

    def get_lr(self):
        return [base_lr * self.gamma**self.last_iter for base_lr in self.base_lrs]


class CosineAnnealingLR(_IterLRScheduler):
    r"""
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class ChainIterLR(object):
    """
    Chain warmup scheduler and large batch scheduler together
    Warmup scheduler is an iteration scheduler, while LargeBatch
    scheduler is an epoch scheduler, we treat both as iteration
    scheduler, i.e. update lr for each iteration.
    """

    def __init__(self,
                 optimizer,
                 cfg_scheduler,
                 world_size,
                 batch_size,
                 data_size,
                 last_epoch):
        lr_scale = world_size * batch_size
        warmup_epochs = cfg_scheduler['warmup_epochs']
        warmup_iters = int(warmup_epochs * data_size)
        self.warmup_scheduler = build_warmup_scheduler(optimizer, warmup_iters, lr_scale)
        self.epoch_scheduler = build_epoch_scheduler(optimizer, cfg_scheduler, data_size)
        self.lr_scheduler = self.warmup_scheduler
        self.last_iter = -warmup_iters
        last_iter = int((last_epoch + 1) * data_size - 1)
        for i in range(-warmup_iters, last_iter + 1):
            self.step()

    def __getattr__(self, attr_name):
        return getattr(self.lr_scheduler, attr_name)

    def step(self):
        if self.last_iter == 0:
            logger.info('warmup done, start Large Batch training')
            self.update_lrs(self.epoch_scheduler)
            self.lr_scheduler = self.epoch_scheduler
        self.last_iter += 1
        self.lr_scheduler.step()

    def update_lrs(self, lr_scheduler):
        optimizer = lr_scheduler.optimizer
        # update base_lrs with lasted lr
        lr_scheduler.base_lrs = list(
            map(lambda group: group['lr'], optimizer.param_groups))


def build_epoch_scheduler(optimizer, cfg, data_size):
    """We treat epoch as iter"""
    cfg['kwargs']['optimizer'] = optimizer
    if cfg['type'] == 'MultiStepLR':
        cfg['kwargs']['milestones'] = [e * data_size for e in cfg['kwargs']['milestones']]
    elif cfg['type'] == 'StepLR':
        cfg['kwargs']['step_size'] = cfg['kwargs']['step_size'] * data_size
    elif cfg['type'] == 'ReduceLROnPlateau':
        cfg['kwargs']['patience'] = cfg['kwargs']['patience'] * data_size
    elif cfg['type'] == 'CosineAnnealingLR':
        cfg['kwargs']['T_max'] = cfg['kwargs']['T_max'] * data_size
    else:
        raise NotImplementedError(f'{cfg} is not supported')
    return build_cls_instance(torch.optim.lr_scheduler, cfg)


def build_warmup_scheduler(optimizer, warmup_iter, lr_scale):
    if warmup_iter < 2:
        gamma = 1.0
    else:
        gamma = lr_scale**(1.0 / (warmup_iter - 1))
    return IterExponentialLR(optimizer, gamma)
