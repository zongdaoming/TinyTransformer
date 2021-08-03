import logging

import torch
from torch.autograd import Function

from .._C import psroi_mask_pooling

logger = logging.getLogger('global')


class PSRoIMaskPoolFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, group_size, spatial_scale, roi_scale, bin_scale, output_dim):
        return g.op(
            "PSRoiMaskPool",
            features,
            rois,
            output_dim_i=output_dim,
            group_size_i=group_size,
            spatial_scale_f=spatial_scale,
            roi_scale_f=roi_scale,
            bin_scale_f=bin_scale)

    @staticmethod
    def forward(self, features, rois, group_size, spatial_scale, roi_scale, bin_scale, output_dim):
        self.save_for_backward(features, rois)
        self.group_size = group_size
        self.spatial_scale = spatial_scale
        self.roi_scale = roi_scale
        self.bin_scale = bin_scale
        self.output_dim = output_dim

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.shape[0]
        output = features.new(num_rois, self.output_dim, self.group_size, self.group_size).zero_()
        mapping_channel = torch.cuda.IntTensor(
            num_rois, self.output_dim, self.group_size, self.group_size).zero_()

        forward_fn = psroi_mask_pooling.forward_cuda
        if not features.is_cuda:
            logger.warning(
                '---CPU version of PSRoIMaskPooling is a dummpy function, which is used to support tocaffe')
            forward_fn = psroi_mask_pooling.forward_cpu

        forward_fn(self.group_size, self.group_size, self.output_dim,
                   self.spatial_scale, self.roi_scale, self.bin_scale,
                   features, rois, output, mapping_channel)
        self.mapping_channel = mapping_channel
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data

        feature, rois = self.saved_tensors
        assert grad_output.is_cuda

        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        psroi_mask_pooling.backward_cuda(
            self.group_size, self.group_size, self.output_dim,
            self.spatial_scale, self.roi_scale, self.bin_scale,
            grad_output, rois, grad_input, self.mapping_channel)
        return grad_input, None, None, None, None, None, None


class PSRoIMaskPool(torch.nn.Module):

    def __init__(self, group_size, roi_scale, bin_scale,
                 output_dim=None, spatial_scale=None):
        super(PSRoIMaskPool, self).__init__()

        self.group_size = int(group_size)
        self.roi_scale = float(roi_scale)
        self.bin_scale = float(bin_scale)

        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated in PSRoIMaskPool.__init__, '
                           'we move `spatial_scale` to forward arguments `stride` '
                           'for flexiability')

        if output_dim is not None:
            logger.warning('`output_dim` is deprecated in PSRoIPool.__ini__, '
                           'we will calculate `output_dim` by chanels of pooled '
                           '`features` and `group_size` dynamically')

    def forward(self, rois, features, stride):
        """
        Arguments:
            rois: [N, >=5] (batch_idx, x1, y1, x2, y2)

        Notes:
            1. rois must be N*5 dim
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :5].contiguous().to(dtype=features.dtype)
        features = features.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        output_dim = features.shape[1] // self.group_size**2
        assert self.group_size**2 * output_dim == features.shape[1]
        return PSRoIMaskPoolFunction.apply(
            features, rois, self.group_size, spatial_scale,
            self.roi_scale, self.bin_scale, output_dim)

    def __repr__(self):
        s = '{name} ({group_size}, {roi_scale} {bin_scale})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        group_size = params['pool_size']
        roi_scale = params['roi_scale']  # default 1.5
        bin_scale = params['bin_scale']  # default 2.0
        return cls(group_size, roi_scale, bin_scale)
