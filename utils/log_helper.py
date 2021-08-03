from __future__ import division

# Standard Library
import logging
import sys
import time
from collections import defaultdict, deque

# Import from third library
import numpy as np
import torch
import torch.distributed as dist


# Import from local
from .deprecate import deprecated_warning
from .environment import default_env as env

logs = set()    

# LOGGER
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


def basicConfig(*args, **kwargs):
    return


# To prevent duplicate logs, we mask this baseConfig setting
logging.basicConfig = basicConfig


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        msg = record.msg
        levelname = record.levelname
        if self.use_color and levelname in COLORS and COLORS[levelname] != WHITE:
            if isinstance(msg, str):
                msg_color = COLOR_SEQ % (30 + COLORS[levelname]) + msg + RESET_SEQ
                record.msg = msg_color
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return

    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)

    logger.addFilter(lambda record: env.is_master())

    format_str = f'%(asctime)s-rk{env.rank}-%(filename)s#%(lineno)d:%(message)s'
    formatter = ColoredFormatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


@deprecated_warning()
def sync_info(output):
    """Sync loss and accuracy across gpus"""

    def filter_fn(x):
        return x.find('loss') >= 0 or x.find('accuracy') >= 0

    output = {name: val.clone() for name, val in output.items() if filter_fn(name)}

    if env.distributed:
        for name, val in output.items():
            if torch.is_tensor(val):
                # link.allreduce(val)
                dist.all_reduce(val)
                output[name] = val / env.world_size

    return {name: val.item() for name, val in output.items()}


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    window_size = 20
    skip_first_k = 0
    precision = 4

    def __init__(self):
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0

    @deprecated_warning(replacement='+=')
    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def __iadd__(self, value):
        if self.skip_first_k > 0:
            self.count = 1
            self.total = value
            self.deque.clear()
            self.deque.append(value)
            self.skip_fist_k -= 1
        else:
            self.deque.append(value)
            self.count += 1
            self.total += value
        return self

    def reset_window_size(self, new_window_size):
        valid_deque = list(deque)[-new_window_size:]
        self.deque = deque(valid_deque, maxlen=new_window_size)

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return np.mean(self.deque)

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    def __str__(self):
        """format:
            latest(global_avg)
        """
        if len(self.deque) == 0:
            return '0(0)'
        else:
            lastest = self.deque[-1]
            global_avg = self.global_avg
            return '{1:.{0}f}({2:.{0}f})'.format(self.precision, lastest, global_avg)


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

    @deprecated_warning(replacement='set_range')
    def set_start_iter(self, start_iter):
        self.start_iter = start_iter

    @deprecated_warning(replacement='set_range')
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
            self.meters[k] += v

    @deprecated_warning()
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

    def __str__(self):
        str_list = ['{}:{}'.format(k, v) for k, v in self.meters.items()]
        return self.delimiter.join(sorted(str_list))


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

default_logger = init_log('global', logging.INFO)
# default_logger = init_log('global', logging.DEBUG)
