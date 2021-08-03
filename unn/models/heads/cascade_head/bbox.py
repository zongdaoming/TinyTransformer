import logging

import torch

from ..utils.bbox_helper import bbox2offset
from ..utils.bbox_helper import clip_bbox
from ..utils.bbox_helper import filter_by_size
from ..utils.bbox_helper import normalize_offset
from ..utils.bbox_helper import offset2bbox
from ..utils.bbox_helper import unnormalize_offset
from ..utils.box_sampler import sample
from ..utils.matcher import match
from ..bbox_head.bbox import predict_bboxes     # noqa F401
from ..bbox_head.bbox import proposal_targets   # noqa F401

logger = logging.getLogger('global')


def proposal_targets_with_gt_flag(proposals, gt_bboxes, image_info, cfg, ignore_regions=None):
    """
    Arguments:
        proposals (FloatTensor, fp32): [N, >=5], (batch_idx, x1, y1, x2, y2, ...)
        gt_bboxes (list of FloatTensor): [B, num_gts, 5], (x1, y1, x2, y2, label)
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
        ignore_regions (list of FloatTensor): None or [B, num_igs, 4] (x1, y1, x2, y2)

    Returns:
        batch_sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)]
        rois (FloatTensor): [R, 5] (batch_idx, x1, y1, x2, y2), sampled rois
        cls_target (LongTensor): [R]
        loc_target (FloatTensor): [R, 4]
        loc_weight (FloatTensor): [R, 4], 1 for positive, otherwise 0
    """
    T = proposals
    B = len(gt_bboxes)
    if ignore_regions is None:
        ignore_regions = [None] * B

    offset_mean = cfg['bbox_normalize']['means']
    offset_std = cfg['bbox_normalize']['stds']

    batch_rois = []
    batch_gt_flags = []
    batch_cls_target = []
    batch_loc_target = []
    batch_loc_weight = []
    batch_sample_record = [None] * B
    for b_ix in range(B):
        rois = proposals[proposals[:, 0] == b_ix]
        if rois.numel() > 0:
            # remove batch idx, score & label
            rois = rois[:, 1:1 + 4]

        # filter gt_bboxes which are too small
        gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)

        # add gts into rois
        gt_flags = rois.new_zeros((rois.shape[0], ), dtype=torch.uint8)
        if gt.numel() > 0:
            rois = torch.cat([rois, gt[:, :4]], dim=0)
            gt_ones = rois.new_ones(gt.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_flags, gt_ones])

        # clip rois which are out of bounds
        rois = clip_bbox(rois, image_info[b_ix])

        rois_target_gt = match(rois, gt, cfg['matcher'], ignore_regions[b_ix])
        pos_inds, neg_inds = sample(rois_target_gt, cfg['sampler'])
        P = pos_inds.numel()
        N = neg_inds.numel()
        if P + N == 0: continue

        # save pos inds and related gts' inds for mask/keypoint head
        sample_record = (pos_inds, rois_target_gt[pos_inds])
        batch_sample_record[b_ix] = sample_record

        # acquire target cls and target loc for sampled rois
        pos_rois = rois[pos_inds]
        neg_rois = rois[neg_inds]
        pos_target_gt = gt[rois_target_gt[pos_inds]]
        gt_flags = gt_flags[pos_inds]
        if pos_target_gt.numel() > 0:
            pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
        else:
            # give en empty tensor if no positive
            pos_cls_target = T.new_zeros((0, ), dtype=torch.int64)
        neg_cls_target = T.new_zeros((N, ), dtype=torch.int64)

        offset = bbox2offset(pos_rois, pos_target_gt)
        pos_loc_target = normalize_offset(offset, offset_mean, offset_std)

        rois = torch.cat([pos_rois, neg_rois], dim=0)
        # for cascade rcnn, add by buxingyuan
        gt_zeros = rois.new_zeros(neg_rois.shape[0], dtype=torch.uint8)
        gt_flags = torch.cat([gt_flags, gt_zeros])
        # end
        b = T.new_full((rois.shape[0], 1), b_ix)
        rois = torch.cat([b, rois], dim=1)
        batch_rois += [rois]
        batch_gt_flags += [gt_flags]
        batch_cls_target += [pos_cls_target, neg_cls_target]
        batch_loc_target += [pos_loc_target, T.new_zeros(N, 4)]

        loc_weight = torch.cat([T.new_ones(P, 4), T.new_zeros(N, 4)])
        batch_loc_weight.append(loc_weight)

    if len(batch_rois) == 0:
        num_rois = 1
        rois = T.new_zeros((num_rois, 5))
        gt_flags = T.new_zeros(num_rois)
        cls_target = T.new_zeros(num_rois).long() - 1  # target cls must be `long` type
        loc_target = T.new_zeros((num_rois, 4))
        loc_weight = T.new_zeros((num_rois, 4))
        logger.warning('no valid proposals, set cls_target to {}'.format(cls_target))
    else:
        rois = torch.cat(batch_rois, dim=0)
        gt_flags = torch.cat(batch_gt_flags, dim=0)
        cls_target = torch.cat(batch_cls_target, dim=0).long()
        loc_target = torch.cat(batch_loc_target, dim=0)
        loc_weight = torch.cat(batch_loc_weight, dim=0)

    return batch_sample_record, rois, cls_target, loc_target, loc_weight, gt_flags


def refine_bboxes(rois, labels, loc_pred, gt_flags, image_info, cfg):
    """
    Something likes predict_bboxes, but this is used in cascade rcnn
    Arguments:
        rois (FloatTensor, fp32): [N, >=5] (batch_ix, x1, y1, x2, y2, ...)
        roi_labels (FloatTensor, fp32): [N]
        loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

    Returns:
        bboxes (FloatTensor): [R, 5], (batch_ix, x1, y1, x2, y2)
    """
    B = len(image_info)

    offset_mean = cfg['bbox_normalize']['means']
    offset_std = cfg['bbox_normalize']['stds']

    if cfg.get('share_location', False):
        offset = loc_pred
    else:
        inds = labels * 4
        inds = torch.stack((inds, inds + 1, inds + 2, inds + 3), 1)
        offset = torch.gather(loc_pred, 1, inds)
    assert offset.size(1) == 4

    offset = unnormalize_offset(offset, offset_mean, offset_std)
    bboxes = offset2bbox(rois[:, 1:1 + 4], offset)

    detected_bboxes = []
    for b_ix in range(B):
        rois_ix = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
        if rois_ix.numel() == 0: continue
        pre_bboxes = bboxes[rois_ix]

        # clip bboxes which are out of bounds
        pre_bboxes[:, :4] = clip_bbox(pre_bboxes[:, :4], image_info[b_ix])

        ix = pre_bboxes.new_full((pre_bboxes.shape[0], 1), b_ix)
        post_bboxes = torch.cat([ix, pre_bboxes], dim=1)
        detected_bboxes.append(post_bboxes)
    detected_bboxes = torch.cat(detected_bboxes, dim=0)

    # filter gt bboxes
    pos_keep = 1 - gt_flags
    keep_inds = torch.nonzero(pos_keep).reshape(-1)
    new_rois = detected_bboxes[keep_inds]

    return new_rois
