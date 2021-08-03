import logging

import torch
from torch.autograd import Function

from .._C import roi_ex_pooling

logger = logging.getLogger('global')


class RoIExPoolFunction(Function):
    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale):
        return g.op(
            "RoiExPool",
            features,
            rois,
            spatial_scale_f=spatial_scale,
            pooled_width_i=pooled_w,
            pooled_height_i=pooled_h)

    @staticmethod
    def forward(self, features, rois, pooled_height, pooled_width, spatial_scale):
        self.save_for_backward(features, rois)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_()
        self.argmax = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_().int()

        assert features.is_contiguous()
        assert rois.is_contiguous()

        forward_fn = roi_ex_pooling.forward_cuda
        if not features.is_cuda:
            logger.warning(
                '---CPU version of RoIPooling is a dummpy function, which is used to support tocaffe')
            forward_fn = roi_ex_pooling.forward_cpu

        forward_fn(pooled_height, pooled_width, spatial_scale, features, rois, output, self.argmax)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data
        feature, rois = self.saved_tensors
        batch_size, num_channels, data_height, data_width = feature.size()
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        assert (grad_output.is_contiguous())
        assert (rois.is_contiguous())

        roi_ex_pooling.backward_cuda(
            self.pooled_height, self.pooled_width, self.spatial_scale,
            grad_output, rois, grad_input, self.argmax)
        return grad_input, None, None, None, None


class RoIExPool(torch.nn.Module):

    def __init__(self, pooled_h, pooled_w, spatial_scale=None):
        super(RoIExPool, self).__init__()

        self.pooled_h = int(pooled_h)
        self.pooled_w = int(pooled_w)

        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated when initializing RoIExPool'
                           'we move spatial_scale to forward arguments `stride` for flexiability')

    def forward(self, rois, feature, stride):
        """
        Arguments:
            rois: [N, >=13] (batch_idx, x1, y1, x2, y2, hx1, hy1, hx2, hy2, ox1, oy1, ox2, oy2)

        Notes:
            1. rois mus
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :13].contiguous().to(dtype=feature.dtype)
        feature = feature.contiguous()
        assert rois.shape[1] == 13, rois.shape
        spatial_scale = 1.0 / stride
        return RoIExPoolFunction.apply(feature, rois, self.pooled_h,
                                     self.pooled_w, spatial_scale)

    def __repr__(self):
        s = '{name} ({pooled_h}, {pooled_w})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        pooled_h = pooled_w = params['pool_size']
        return cls(pooled_h, pooled_w)
