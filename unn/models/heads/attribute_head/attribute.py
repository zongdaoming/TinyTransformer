import logging

import torch

from ....extensions import nms
from ..utils.bbox_helper import filter_by_size
from ..utils.box_sampler import sample
from ..utils.matcher import match

logger = logging.getLogger('global')


def proposal_targets(proposals, gt_bboxes, gt_attris, image_info, cfg, ignore_regions=None):
    T = proposals
    B = len(gt_bboxes)
    C = gt_attris[0].shape[1]
    if ignore_regions is None:
        ignore_regions = [None] * B

    batch_rois = []
    batch_cls_target = []
    batch_sample_record = [None] * B
    for b_ix in range(B):
        rois = proposals[proposals[:, 0] == b_ix].float()
        if rois.numel() > 0:
            # remove batch idx, score & label
            rois = rois[:, 1:1 + 4]
        gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)
        batch_gt_attris = gt_attris[b_ix]
    
        if gt.numel() > 0:
            rois = torch.cat([rois, gt[:, :4]], dim=0)
        rois_target_gt = match(rois, gt, cfg['matcher'], ignore_regions[b_ix])
        pos_inds, neg_inds = sample(rois_target_gt, cfg['sampler'])
        P = pos_inds.numel()
        N = neg_inds.numel()
        if P + N == 0: continue     # noqa E701

        # save pos inds and related gts' inds for mask/keypoint head
        sample_record = (pos_inds, rois_target_gt[pos_inds])
        batch_sample_record[b_ix] = sample_record

        # acquire target cls and target loc for sampled rois
        pos_rois = rois[pos_inds]
        neg_rois = rois[neg_inds]
        pos_target_gt = batch_gt_attris[rois_target_gt[pos_inds]]
        if pos_target_gt.numel() > 0:
            pos_cls_target = pos_target_gt.to(dtype=torch.int64)
        else:
            # give en empty tensor if no positive
            pos_cls_target = T.new_zeros((0, ), dtype=torch.int64)
        neg_cls_target = T.new_zeros((N, C), dtype=torch.int64)

        rois = torch.cat([pos_rois, neg_rois], dim=0)
        b = T.new_full((rois.shape[0], 1), b_ix)
        rois = torch.cat([b, rois], dim=1)
        batch_rois += [rois]
        batch_cls_target += [pos_cls_target, neg_cls_target]

    rois = torch.cat(batch_rois, dim=0)
    cls_target = torch.cat(batch_cls_target, dim=0).long()

    return batch_sample_record, rois, cls_target
