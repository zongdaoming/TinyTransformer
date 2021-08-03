import logging

import torch
from .bbox_helper import bbox_iou_overlaps
from .bbox_helper import bbox_iof_overlaps

logger = logging.getLogger('global')


def match(bboxes, gt, cfg, gt_ignores=None):
    """
    Match roi to gt

    Temporarily used tensors:
        overlaps (FloatTensor): [N, M], ious of dt(N) with gt(M)
        ignore_overlaps (FloatTensor): [N, K], ious of dt(N) with ignore regions(K)

    Returns:
        target (LongTensor): [N], matched gt index for each roi.
                            1. if a roi is positive, it's target is matched gt index (>=0)
                            2. if a roi is negative, it's target is -1,
                            3. if a roi isn't positive nor negative, it's target is -2;
    """
    NEGATIVE_TARGET = -1
    IGNORE_TARGET = -2
    N = bboxes.shape[0]
    M = gt.shape[0]

    # check M > 0 for no-gt support
    overlaps = bbox_iou_overlaps(bboxes, gt) if M > 0 else bboxes.new_zeros(N, 1)
    ignore_overlaps = None
    if gt_ignores is not None and gt_ignores.numel() > 0:
        ignore_overlaps = bbox_iof_overlaps(bboxes, gt_ignores)

    target = bboxes.new_full((N, ), IGNORE_TARGET, dtype=torch.int64)
    dt_to_gt_max, dt_to_gt_argmax = overlaps.max(dim=1)

    # rule 1: negative if maxiou < negative_iou_thresh:
    neg_mask = dt_to_gt_max < cfg['negative_iou_thresh']
    target[neg_mask] = NEGATIVE_TARGET

    # rule 2: positive if maxiou > pos_iou_thresh
    pos_mask = dt_to_gt_max > cfg['positive_iou_thresh']
    target[pos_mask] = dt_to_gt_argmax[pos_mask]

    # rule 3: positive if a dt has highest iou with any gt
    if cfg.get('allow_low_quality_match') and M > 0:
        overlaps = overlaps.t()  # IMPORTANT, for faster caculation
        gt_to_dt_max, _ = overlaps.max(dim=1)
        dt_gt_pairs = torch.nonzero((overlaps >= gt_to_dt_max[:, None] - 1e-3))
        if dt_gt_pairs.numel() > 0:
            lqm_dt_inds = dt_gt_pairs[:, 1]
            target[lqm_dt_inds] = dt_to_gt_argmax[lqm_dt_inds]
            pos_mask[lqm_dt_inds] = 1

    # rule 4: dt has high iou with ignore regions may not supposed to be negative
    if ignore_overlaps is not None and ignore_overlaps.numel() > 0:
        dt_to_ig_max, _ = ignore_overlaps.max(dim=1)
        ignored_dt_mask = dt_to_ig_max > cfg['ignore_iou_thresh']
        # remove positives from ignored
        ignored_dt_mask = (ignored_dt_mask ^ (ignored_dt_mask & pos_mask))
        target[ignored_dt_mask] = IGNORE_TARGET
    return target
