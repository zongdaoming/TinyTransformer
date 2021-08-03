# Standard Library
import functools
import inspect
import logging
import os


def deprecated_warning(replacement=None, message=None):
    frame = inspect.stack()[1][0]
    info = inspect.getframeinfo(frame)
    filename = os.path.basename(info.filename)
    lineno = info.lineno

    def decorator(func):
        position = f'function "{func.__name__}@{filename}#{lineno}"'

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger('global')
            if message is not None:
                logger.warning(message)
            elif replacement is not None:
                logger.warning(f"{position} will be DEPRECATED soon, "
                               f"please use {replacement} instead")
            else:
                logger.warning(f"{position} will be DEPRECATED soon")
            return func(*args, **kwargs)

        return wrapper

    return decorator
