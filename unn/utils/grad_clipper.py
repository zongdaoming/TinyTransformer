import logging

from torch.nn.utils import clip_grad_norm

logger = logging.getLogger('global')


class GradClipper(object):
    """
    Clip gradients by
    1. provided max_norm
    2. average total_norm from watch_iter iterations
    """

    def __init__(self, max_norm=0, norm_type=2, watch_iter=0):
        assert (max_norm > 0 and watch_iter == 0) or (max_norm == 0 and watch_iter > 0)
        self.watch_iter = watch_iter
        self.last_iter = 0
        self.max_norm = max_norm
        self.norm_type = norm_type

    def update_max_norm(self, parameters):
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type)
            total_norm += param_norm**self.norm_type
        total_norm = total_norm**(1. / self.norm_type)
        logger.info('total_norm:{}'.format(total_norm))
        self.max_norm += total_norm
        self.last_iter += 1

    def clip_grad(self, parameters):
        parameters = [p for p in parameters if p.requires_grad and p.grad is not None]
        if self.last_iter < self.watch_iter:
            self.update_max_norm(parameters)
            if self.last_iter == self.watch_iter:
                self.max_norm /= self.watch_iter
                logger.info('max_norm:{}'.format(self.max_norm))
        else:
            clip_grad_norm(parameters, self.max_norm, self.norm_type)
