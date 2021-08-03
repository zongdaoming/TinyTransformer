# Standard Library
import inspect
import os
from collections import defaultdict

# Import from local
from .dist_helper import MASTER_RANK, get_local_rank, get_rank, get_world_size


class Environment(object):
    """A namespace that used to transfer variables across modules
    """

    def __init__(self):
        self.__trace__ = defaultdict(list)
        self._keep_value = False

    @property
    def rank(self):
        self._rank = get_rank()
        return self._rank
        # if not hasattr(self, '_rank'):
        #     self._rank = get_rank()
        # return self._rank

    @property
    def local_rank(self):
        self._local_rank = get_local_rank()
        return self._local_rank
        # if not hasattr(self, '_local_rank'):
        #     self._local_rank = get_local_rank()
        # return self._local_rank

    @property
    def world_size(self):
        self._world_size = get_world_size()
        return self._world_size
        # if not hasattr(self, '_world_size'):
        #     self._world_size = get_world_size()
        # return self._world_size

    def is_master(self):
        return self.rank == MASTER_RANK

    @property
    def distributed(self):
        return self.world_size > 1

    def keep_value(self, keep=True):
        self._keep_value = keep

    def __setattr__(self, attr, value):
        frame = inspect.stack()[1][0]
        info = inspect.getframeinfo(frame)
        filename = os.path.basename(info.filename)
        lineno = info.lineno

        if not attr.startswith('_'):
            if self._keep_value:
                self.__trace__[attr].append((filename, lineno, value))
            else:
                self.__trace__[attr].append((filename, lineno))

        self.__dict__[attr] = value

    def format(self, info_tuple):
        if len(info_tuple) == 3:
            return '{}#{}:{}'.format(info_tuple[0], info_tuple[1], info_tuple[2])
        elif len(info_tuple) == 2:
            return '{}#{}'.format(info_tuple[0], info_tuple[1])
        else:
            raise ValueError

    def trace(self, attr):
        return [self.format(info) for info in self.__trace__[attr]]

    def trace_all(self):
        return {attr: self.trace(attr) for attr in sorted(self.__dict__.keys())}


default_env = Environment()


if __name__ == '__main__':
    default_env.seed = 131
    default_env.deterministic = False
    print(default_env.trace('seed'))
    print(default_env.trace('deterministic'))
    print(default_env.trace_all())
    default_env.seed = 132
    default_env.deterministic = True
    print(default_env.trace('seed'))
    print(default_env.trace('deterministic'))
    print(default_env.trace_all())
