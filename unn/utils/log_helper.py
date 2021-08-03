from __future__ import division

import os
import time
import logging
from collections import defaultdict
from collections import deque

import torch
import linklink as link

from .dist_helper import get_world_size

logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return

    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)

    rank = 0
    if 'SLURM_PROCID' in os.environ:
        # only print log for rank 0
        rank = int(os.environ['SLURM_PROCID'])
        logger.addFilter(lambda record: rank == 0)

    format_str = f'%(asctime)s-rk{rank}-%(filename)s#%(lineno)d:%(message)s'
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def sync_info(output):
    """Sync loss and accuracy across gpus"""

    def filter_fn(x):
        return x.find('loss') >= 0 or x.find('accuracy') >= 0

    output = {name: val.clone() for name, val in output.items() if filter_fn(name)}

    world_size = get_world_size()
    if world_size > 1:
        for name, val in output.items():
            if torch.is_tensor(val):
                link.allreduce(val)
                output[name] = val / world_size

    return {name: val.item() for name, val in output.items()}


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    window_size = 20

    def __init__(self):
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t", cur_iter=0, start_iter=0):
        self.meters = defaultdict(SmoothedValue)  # no instantiation here
        self.first_iter_flag = True
        self.start_iter = start_iter
        self.delimiter = delimiter
        self.cur_iter = cur_iter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def set_window_size(self, window_size):
        SmoothedValue.window_size = window_size

    def set_start_iter(self, start_iter):
        self.start_iter = start_iter

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def update(self, detail_time={}, **kwargs):
        # As batch time of the first iter is much longer than normal time, we
        # exclude the first iter for more accurate speed statistics. If the window
        # size is 1, batch time and loss of the first iter will display, but will not
        # contribute to the global average data.
        if self.first_iter_flag and self.start_iter + 1 == self.cur_iter:
            self.first_iter_flag = False
            for name, meter in self.meters.items():
                meter.count -= 1
                meter.total -= meter.deque.pop()

        kwargs.update(detail_time)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def get_str(self, arg="timer"):
        if arg == "timer":
            time_str = []
            for name, meter in self.meters.items():
                if name.endswith("_time"):
                    time_str.append("{}:{:.4f}({:.4f})".format(name, meter.avg, meter.global_avg))
            return self.delimiter.join(time_str)
        elif arg == "loss":
            loss_dict = defaultdict(list)
            for name, meter in self.meters.items():
                if not name.endswith("time"):
                    prefix, loss_name = name.split('.', 1)
                    loss_dict[prefix].append('{}:{:.4f}'.format(loss_name, meter.avg))
            loss_str = ['{}({})'.format(prefix, ' '.join(val)) for prefix, val in loss_dict.items()]
            return self.delimiter.join(sorted(loss_str))
        else:
            time_str = []
            for name, meter in self.meters.items():
                if name.endswith(f"_{arg}"):
                    time_str.append("{}:{:.4f}({:.4f})".format(name, meter.avg, meter.global_avg))
            return self.delimiter.join(time_str)
            # raise KeyError("keyword {} not supported".format(arg))


def get_cur_time():
    torch.cuda.synchronize()
    return time.time()


def get_diff_time(pre_time):
    torch.cuda.synchronize()
    diff_time = time.time() - pre_time
    return diff_time, time.time()


def add_diff_time(base_time, pre_time):
    torch.cuda.synchronize()
    base_time += time.time() - pre_time
    return base_time, time.time()


meters = MetricLogger(delimiter=" ")
timer = MetricLogger(delimiter=" ")
