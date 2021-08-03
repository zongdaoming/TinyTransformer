import logging

import torch
from torch.autograd import Function

from .._C import roi_align

logger = logging.getLogger('global')


# TODO use save_for_backward instead
class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio):
        return g.op(
            "RoiAlign",
            features,
            rois,
            spatial_scale_f=spatial_scale,
            pooled_width_i=pooled_w,
            pooled_height_i=pooled_h,
            sample_num_i=sampling_ratio)

    @staticmethod
    def forward(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio):
        self.save_for_backward(features, rois)
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, pooled_h, pooled_w).zero_()
        assert features.is_contiguous() and rois.is_contiguous()

        forward_fn = roi_align.forward_cuda
        if not features.is_cuda:
            logger.warning(
                '---CPU version of RoIAlignPooling is a dummpy function, which is used to support tocaffe')
            forward_fn = roi_align.forward_cpu
        forward_fn(pooled_h, pooled_w, spatial_scale, sampling_ratio, features, rois, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data
        feature, rois = self.saved_tensors
        grad_output = grad_output.contiguous()
        feature = feature.contiguous()
        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = feature.new(batch_size, num_channels, data_height, data_width).zero_()
        assert(grad_output.is_contiguous())
        assert(rois.is_contiguous())

        backward_fn = roi_align.backward_cuda
        backward_fn(
            self.pooled_h, self.pooled_w, self.spatial_scale,
            self.sampling_ratio, grad_output, rois, grad_input)
        return grad_input, None, None, None, None, None


class RoIAlignPool(torch.nn.Module):

    def __init__(self, pooled_h, pooled_w, sampling_ratio, spatial_scale=None):
        super(RoIAlignPool, self).__init__()

        self.pooled_w = int(pooled_w)
        self.pooled_h = int(pooled_h)
        self.sampling_ratio = int(sampling_ratio)

        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated when initializing RoIAlignPool'
                           'we move spatial_scale to forward arguments `stride` for flexiability')

    def forward(self, rois, feature, stride):
        """
        Arguments:
            rois: [N, >=5] (batch_idx, x1, y1, x2, y2)

        Notes:
            1. rois must be N*5 dim
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :5].contiguous().to(dtype=feature.dtype)
        feature = feature.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        return RoIAlignFunction.apply(
            feature, rois, self.pooled_h, self.pooled_w, spatial_scale, self.sampling_ratio)

    def __repr__(self):
        s = '{name} ({pooled_h}, {pooled_w} {sampling_ratio})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        pooled_h = pooled_w = params['pool_size']
        sampling_ratio = params['sampling_ratio']
        return cls(pooled_h, pooled_w, sampling_ratio)
