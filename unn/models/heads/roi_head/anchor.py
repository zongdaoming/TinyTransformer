import logging

import torch

from ....extensions import nms
from ..utils.bbox_helper import bbox2offset
from ..utils.bbox_helper import clip_bbox
from ..utils.bbox_helper import filter_by_size
from ..utils.bbox_helper import offset2bbox
from ..utils.box_sampler import sample
from ..utils.matcher import match

logger = logging.getLogger('global')


def anchor_targets(all_anchors, gt_bboxes, image_info, cfg, ignore_regions=None):
    """
    Match anchors with gt bboxes and sample batch samples for training

    Arguments:
        all_anchors (FloatTensor, fp32): [K, 4], for a single layer in FPN, k_i = H * W * A, K = sum(k_i)
        gt_bboxes (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
        ignore_regions (list of FloatTensor): None or [B, I, 4] (x1, y1, x2, y2)

    Returns:
        cls_target (LongTensor): [B, K], {-1, 0, 1} for RPN, {-1, 0, 1...C} for retinanet
        loc_target (FloatTensor): [B, K, 4]
        sample_cls_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0
        sample_loc_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0  # only positive samples
    """
    B = len(gt_bboxes)
    K = all_anchors.shape[0]
    if ignore_regions is None:
        ignore_regions = [None] * B

    cls_target = all_anchors.new_full((B, K), -1, dtype=torch.int64)
    loc_target = all_anchors.new_zeros((B, K, 4))
    sample_cls_mask = all_anchors.new_zeros((B, K), dtype=torch.uint8)
    sample_loc_mask = all_anchors.new_zeros((B, K), dtype=torch.uint8)

    for b_ix in range(B):
        # filter gt bboxes which are too small
        gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)

        # filter anchors which are out of bounds
        img_h, img_w = image_info[b_ix][:2]
        border = cfg['allowed_border']
        if border >= 0:
            inside_mask = ((all_anchors[:, 0] > -border)
                           & (all_anchors[:, 1] > -border)
                           & (all_anchors[:, 2] < img_w + border)
                           & (all_anchors[:, 3] < img_h + border))
            inside_anchors = all_anchors[inside_mask]
            inside_inds = torch.nonzero(inside_mask).reshape(-1)
        else:
            inside_inds = torch.arange(K, dtype=torch.int64, device=all_anchors.device)
            inside_anchors = all_anchors

        anchor_target_gt = match(inside_anchors, gt, cfg['matcher'], ignore_regions[b_ix])

        pos_inds, neg_inds = sample(anchor_target_gt, cfg['sampler'])

        # acquire target cls and loc for sampled anchors
        pos_anchors = inside_anchors[pos_inds]
        pos_target_gt = gt[anchor_target_gt[pos_inds]]
        pos_loc_target = bbox2offset(pos_anchors, pos_target_gt)
        if pos_target_gt.numel() > 0:
            pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
        else:
            pos_cls_target = pos_target_gt.new_zeros((0, ), dtype=torch.int64)
            pos_loc_target = pos_target_gt.new_zeros((0, 4), dtype=torch.int64).float()

        cls_target[b_ix, inside_inds[pos_inds]] = pos_cls_target
        loc_target[b_ix, inside_inds[pos_inds]] = pos_loc_target
        cls_target[b_ix, inside_inds[neg_inds]] = 0

        sample_cls_mask[b_ix, inside_inds[pos_inds]] = 1
        sample_cls_mask[b_ix, inside_inds[neg_inds]] = 1
        sample_loc_mask[b_ix, inside_inds[pos_inds]] = 1

    return cls_target, loc_target, sample_cls_mask, sample_loc_mask


def predict_rois(anchors, cls_pred, loc_pred, image_info, cfg):
    """
    Arguments:
        anchors (FloatTensor, fp32): [K, 4]
        cls_pred (FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
        loc_pred (FloatTensor, fp32): [B, K, 4]
        image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

    Returns:
        rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
    """
    B, K, C = cls_pred.shape
    roi_min_size = cfg['roi_min_size']
    pre_nms_top_n = cfg['pre_nms_top_n']
    post_nms_top_n = cfg['post_nms_top_n']
    pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
    score_thresh = cfg['pre_nms_score_thresh'] if cls_pred.shape[1] > 120 else 0

    concat_anchors = torch.stack([anchors.clone() for _ in range(B)])
    rois = offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4)

    batch_rois = []
    for b_ix in range(B):
        # clip rois and filter rois which are too small
        image_rois = rois[b_ix]
        image_rois = clip_bbox(image_rois, image_info[b_ix])
        image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
        image_cls_pred = cls_pred[b_ix][filter_inds]
        if image_rois.numel() == 0: continue  # noqa E701

        for cls in range(C):
            cls_rois = image_rois
            scores = image_cls_pred[:, cls]
            if score_thresh > 0:
                # to reduce computation
                keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                if keep_idx.numel() == 0: continue  # noqa E701
                cls_rois = cls_rois[keep_idx]
                scores = scores[keep_idx]

            # do nms per image, only one class
            _pre_nms_top_n = min(pre_nms_top_n, scores.shape[0])
            scores, order = scores.topk(_pre_nms_top_n, sorted=True)
            cls_rois = cls_rois[order, :]
            cls_rois = torch.cat([cls_rois, scores[:, None]], dim=1)

            cls_rois, keep_idx = nms(cls_rois, cfg['nms'])
            if post_nms_top_n > 0:
                cls_rois = cls_rois[:post_nms_top_n]

            ix = cls_rois.new_full((cls_rois.shape[0], 1), b_ix)
            c = cls_rois.new_full((cls_rois.shape[0], 1), cls + 1)
            cls_rois = torch.cat([ix, cls_rois, c], dim=1)
            batch_rois.append(cls_rois)

    if len(batch_rois) == 0:
        return anchors.new_zeros((1, 7))
    return torch.cat(batch_rois, dim=0)
