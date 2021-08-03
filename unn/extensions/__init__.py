# flake8: noqa
import logging

import numpy as np
import torch

from .python.deformable_conv import conv_offset2d
from .python.deformable_conv import DeformableConv
from .python.deformable_conv import DeformableConvInOne
from .python.deformable_conv import DeformConv2d
from .python.focal_loss import SigmoidFocalLossFunction
from .python.focal_loss import SoftmaxFocalLossFunction
from .python.iou_overlap import gpu_iou_overlap
from .python.nms import naive_nms
from .python.nms import soft_nms
from .python.psroi_mask_pool import PSRoIMaskPool
from .python.psroi_pool import PSRoIPool
from .python.roi_align import RoIAlignPool
from .python.roi_pool import RoIPool

logger = logging.getLogger('global')


def build_generic_roipool(pool_cfg):
    """
    Construct generic RoIPooling-like class

    Arguments:
        pool_cfg: `method` key to speicify roi_pooling method,
            other keys is configuration for this roipooling `method`
    """
    ptype = pool_cfg['method']
    pool_cls = {
        'psroipool': PSRoIPool,
        'psroimaskpool': PSRoIMaskPool,
        'roialignpool': RoIAlignPool,
        'roipool': RoIPool
    }[ptype]
    return pool_cls.from_params(pool_cfg)


class MultiLevelGenericRoIPooling(torch.nn.Module):
    """wrapper for multi-level roi feature extractor"""

    def __init__(self, pool_cfg, levels):
        """
        Arguments:
            pool_cfg (dict): config of roipooling-like method
            levels (list of int): indices of used levels of input features.
            (e.g. in FPN, input features are P2~P6, but we pooling P2~P5 only, in which case levels=[0,1,2,3])
        """
        super(MultiLevelGenericRoIPooling, self).__init__()
        self.roi_feature_extractor = build_generic_roipool(pool_cfg)
        self.levels = levels

    def forward(self, rois, features, strides):
        """
        Unified interface for multi-level (e.g. FPN) roi pooling-like method

        Arguments:
            rois (list): list of rois of each level
            features (list): feature map of each level
            strides (list): stride of feature map of each level
            levels (list): indices of features to pooling
        """
        assert isinstance(rois, list)
        assert isinstance(features, list)
        assert isinstance(strides, list)
        assert len(features) == len(strides)

        output = []
        for lvl_idx in self.levels:
            if rois[lvl_idx].numel() > 0:
                lvl_feat = features[lvl_idx]
                lvl_rois = rois[lvl_idx]
                lvl_stride = strides[lvl_idx]
                output.append(self.roi_feature_extractor(lvl_rois, lvl_feat, lvl_stride))
        return torch.cat(output, dim=0)


def nms(bboxes, cfg, descending=True):
    """
    Arguments:
        bboxes (FloatTensor): [N, 5] (x1,y1,x2,y2,score), sorted by score in descending order
        cfg (dict): config for nms
    """
    bboxes = bboxes.contiguous()
    assert bboxes.numel() > 0
    assert bboxes.shape[1] == 5
    if not descending:
        sorted_scores, order = torch.sort(bboxes[:, 4], descending=True)
        bboxes = bboxes[order]

    if cfg['type'] == 'naive':
        if cfg['nms_iou_thresh'] < 0:
            keep_index = torch.arange(bboxes.shape[0], device=bboxes.device)
            nmsed_bboxes = bboxes
        else:
            keep_index = naive_nms(bboxes, cfg['nms_iou_thresh'])
            nmsed_bboxes = bboxes[keep_index]
    elif cfg['type'] == 'soft':
        sigma = np.float32(cfg.get('softnms_sigma', 0.5))
        Nt = np.float32(cfg['nms_iou_thresh'])
        threshold = np.float32(cfg.get('softnms_bbox_score_thresh', 0.0001))
        method = cfg.get('softnms_method', 'linear')
        methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
        assert method in methods, 'Unknown soft_nms method: {}'.format(method)
        method = methods[method]
        if Nt < 0:
            keep_index = torch.arange(bboxes.shape[0], device=bboxes.device)
            nmsed_bboxes = bboxes
        else:
            nmsed_bboxes, keep_index = soft_nms(bboxes.cpu().numpy(), sigma, Nt, threshold, method)
            keep_index = torch.from_numpy(keep_index).to(dtype=torch.int64, device=bboxes.device)
            nmsed_bboxes = torch.from_numpy(nmsed_bboxes).to(bboxes)
    else:
        raise NotImplementedError('NMS method {} not supported'.format(cfg['type']))
    if not descending:
        keep_index = order[keep_index]
    return nmsed_bboxes, keep_index
