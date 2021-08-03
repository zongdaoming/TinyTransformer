import logging

import torch

from ....extensions import nms
from ..utils.bbox_helper import bbox2offset
from ..utils.bbox_helper import clip_bbox
from ..utils.bbox_helper import filter_by_size
from ..utils.bbox_helper import normalize_offset
from ..utils.bbox_helper import offset2bbox
from ..utils.bbox_helper import unnormalize_offset
from ..utils.bbox_helper import box_voting
from ..utils.bbox_helper import offset2tiled_bbox
from ..utils.bbox_helper import clip_tiled_boxes
from ..utils.bbox_helper import flip_tiled_bboxes
from ..utils.box_sampler import sample
from ..utils.matcher import match

logger = logging.getLogger('global')


def proposal_targets(proposals, gt_bboxes, image_info, cfg, ignore_regions=None):
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
        if gt.numel() > 0:
            rois = torch.cat([rois, gt[:, :4]], dim=0)

        # clip rois which are out of bounds
        rois = clip_bbox(rois, image_info[b_ix])

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
        pos_target_gt = gt[rois_target_gt[pos_inds]]
        if pos_target_gt.numel() > 0:
            pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
        else:
            # give en empty tensor if no positive
            pos_cls_target = T.new_zeros((0, ), dtype=torch.int64)
        neg_cls_target = T.new_zeros((N, ), dtype=torch.int64)

        offset = bbox2offset(pos_rois, pos_target_gt)
        pos_loc_target = normalize_offset(offset, offset_mean, offset_std)

        rois = torch.cat([pos_rois, neg_rois], dim=0)
        b = T.new_full((rois.shape[0], 1), b_ix)
        rois = torch.cat([b, rois], dim=1)
        batch_rois += [rois]
        batch_cls_target += [pos_cls_target, neg_cls_target]
        batch_loc_target += [pos_loc_target, T.new_zeros(N, 4)]

        loc_weight = torch.cat([T.new_ones(P, 4), T.new_zeros(N, 4)])
        batch_loc_weight.append(loc_weight)

    if len(batch_rois) == 0:
        num_rois = 1
        rois = T.new_zeros((num_rois, 5))
        cls_target = T.new_zeros(num_rois).long() - 1  # target cls must be `long` type
        loc_target = T.new_zeros((num_rois, 4))
        loc_weight = T.new_zeros((num_rois, 4))
        logger.warning('no valid proposals, set cls_target to {}'.format(cls_target))
    else:
        rois = torch.cat(batch_rois, dim=0)
        cls_target = torch.cat(batch_cls_target, dim=0).long()
        loc_target = torch.cat(batch_loc_target, dim=0)
        loc_weight = torch.cat(batch_loc_weight, dim=0)

    return batch_sample_record, rois, cls_target, loc_target, loc_weight


def predict_bboxes(rois, cls_pred, loc_pred, image_info, cfg):
    """
    Arguments:
        rois (FloatTensor, fp32): [N, >=5] (batch_ix, x1, y1, x2, y2, ...)
        cls_pred (FloatTensor, fp32): [N, C], C is num_classes, including background class
        loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

    Returns:
        bboxes (FloatTensor): [R, 7], (batch_ix, x1, y1, x2, y2, score, cls)
    """
    offset_mean = cfg['bbox_normalize']['means']
    offset_std = cfg['bbox_normalize']['stds']
    bbox_score_thresh = max(cfg['bbox_score_thresh'], 0)
    share_location = cfg.get('share_location', False)
    num_top_n = cfg.get('top_n', -1)

    # acquire all predicted bboxes
    B, (R, C) = len(image_info), cls_pred.shape
    loc_pred = unnormalize_offset(loc_pred.view(-1, 4), offset_mean, offset_std)
    bboxes = offset2bbox(rois[:, 1:1 + 4], loc_pred.view(R, -1))
    bboxes = torch.cat([bboxes for _ in range(C)], dim=1) if share_location else bboxes
    bboxes = torch.cat([bboxes.view(-1, 4), cls_pred.view(-1, 1)], dim=1).view(R, -1)

    detected_bboxes = []
    detected_scores = []
    for b_ix in range(B):
        img_inds = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
        if len(img_inds) == 0: continue     # noqa E701
        img_bboxes = bboxes[img_inds]
        img_scores = cls_pred[img_inds]

        # clip bboxes which are out of bounds
        img_bboxes = img_bboxes.view(-1, 5)
        img_bboxes[:, :4] = clip_bbox(img_bboxes[:, :4], image_info[b_ix])
        img_bboxes = img_bboxes.view(-1, C * 5)

        inds_all = img_scores > bbox_score_thresh
        result_bboxes, result_scores = [], []
        for cls in range(1, C):
            # keep well-predicted bbox
            inds_cls = inds_all[:, cls].nonzero().reshape(-1)
            img_bboxes_cls = img_bboxes[inds_cls, cls * 5:(cls + 1) * 5]
            img_scores_cls = img_scores[inds_cls]
            if len(img_bboxes_cls) == 0: continue   # noqa E701

            # do nms, can choose the nms type, naive or softnms
            order = img_bboxes_cls[:, 4].sort(descending=True)[1]
            img_bboxes_cls = img_bboxes_cls[order]
            img_scores_cls = img_scores_cls[order]
            if cfg.get('cls_nms', None) is not None:
                cfg['nms']['nms_iou_thresh'] = cfg['cls_nms'][cls - 1]
            img_bboxes_cls, keep_inds = nms(img_bboxes_cls, cfg['nms'])
            img_scores_cls = img_scores_cls[keep_inds]
            if cfg.get('post_bbox_score_thresh', None) is not None:
                keep = img_bboxes_cls[:, 4] > cfg['post_bbox_score_thresh'][cls - 1]
                img_bboxes_cls = img_bboxes_cls[keep]
                img_scores_cls = img_scores_cls[keep]

            ix = img_bboxes_cls.new_full((img_bboxes_cls.shape[0], 1), b_ix)
            c = img_bboxes_cls.new_full((img_bboxes_cls.shape[0], 1), cls)
            result_bboxes.append(torch.cat([ix, img_bboxes_cls, c], dim=1))
            result_scores.append(img_scores_cls)
        if len(result_bboxes) == 0: continue  # noqa E701

        # keep the top_n well-predicted bboxes per image
        result_bboxes = torch.cat(result_bboxes, dim=0)
        result_scores = torch.cat(result_scores, dim=0)
        if num_top_n >= 0 and num_top_n < result_bboxes.shape[0]:
            # _, topk_inds = result_bboxes[:, 5].topk(num_top_n)
            # result_bboxes = result_bboxes[topk_inds]
            # result_scores = result_scores[topk_inds]
            num_base = result_bboxes.shape[0] - num_top_n + 1
            thresh = torch.kthvalue(result_bboxes[:, 5].cpu(), num_base)[0]
            keep_inds = result_bboxes[:, 5] >= thresh.item()
            result_bboxes = result_bboxes[keep_inds][:num_top_n]
            result_scores = result_scores[keep_inds][:num_top_n]
        detected_bboxes.append(result_bboxes)
        detected_scores.append(result_scores)

    if len(detected_bboxes) == 0:
        return rois.new_zeros((1, 7)), rois.new_zeros((1, C))
    return torch.cat(detected_bboxes, dim=0), torch.cat(detected_scores, dim=0)


def acquire_predicted_bboxes(rois, cls_pred, loc_pred, image_info, cfg):
    """
    Arguments:
        rois (FloatTensor, fp32): [N, >=5] (batch_ix, x1, y1, x2, y2, ...)
        cls_pred (FloatTensor, fp32): [N, C], C is num_classes, including background class
        loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

    Returns:
        cls_pred (FloatTensor, fp32): [N, C], C is num_classes, including background class
        pred_bboxes (FloatTensor): [N, C*4] or [N, 4]

    acquire_predicted_bboxes + bboxes_results_with_nms_and_limit = predict_bboxes
    """
    B = len(image_info)  # supposed to be 1
    N, C = cls_pred.shape[:2]
    assert (N == rois.shape[0])

    if B > 1:
        raise NotImplementedError('Batch > 1 not supported in multi-scale test yet')
    b_ix = 0
    offset_mean = cfg['bbox_normalize']['means']  # noqa F841
    offset_std = cfg['bbox_normalize']['stds']
    offset_inv_std = [1 / std for std in offset_std]

    # acquire all predicted bboxes
    offset = loc_pred
    pred_bboxes = offset2tiled_bbox(rois[:, 1:1 + 4], offset, offset_inv_std)
    pred_bboxes = clip_tiled_boxes(pred_bboxes, image_info[b_ix])

    if image_info[b_ix][5]:  # is_flip
        pred_bboxes = flip_tiled_bboxes(pred_bboxes, image_info[b_ix][1])

    img_resize_scale = image_info[b_ix][2]
    pred_bboxes = pred_bboxes / img_resize_scale
    return cls_pred, pred_bboxes


def bboxes_results_with_nms_and_limit(cls_pred, loc_pred, image_info, cfg):
    """
    Arguments:
        cls_pred (FloatTensor, fp32): [N, C], C is num_classes, including background class
        loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

    Returns:
        bboxes (FloatTensor): [R, 7], (batch_ix, x1, y1, x2, y2, score, cls)

    acquire_predicted_bboxes + bboxes_results_with_nms_and_limit = predict_bboxes
    """
    B = len(image_info)
    N, C = cls_pred.shape[:2]

    if B > 1:
        raise NotImplementedError('Batch > 1 not supported in multi-scale test yet')

    b_ix = 0
    detected_bboxes = []
    for cls in range(1, C):
        # acquire all predicted bboxes
        scores = cls_pred[:, cls]
        if cfg.get('share_location', False):
            bboxes = loc_pred
        else:
            bboxes = loc_pred[:, cls * 4:cls * 4 + 4]
        pre_bboxes = torch.cat([bboxes, scores[:, None]], dim=1)

        # keep well-predicted bbox
        if cfg['bbox_score_thresh'] > 0:
            keep = pre_bboxes[:, 4] > cfg['bbox_score_thresh']
            pre_bboxes = pre_bboxes[keep]
        if pre_bboxes.numel() == 0: continue  # noqa E701

        # do nms, can choose the nms type, naive or softnms
        _, order = pre_bboxes[:, 4].sort(descending=True)
        pre_bboxes = pre_bboxes[order, :]
        _boxes, _indices = nms(pre_bboxes, cfg['nms'])

        # refine the post-NMS boxes using bounding-box voting
        if cfg.get('bbox_vote', None):
            _boxes = box_voting(
                _boxes,
                pre_bboxes,
                cfg['bbox_vote'].get('vote_th', 0.9),
                scoring_method=cfg['bbox_vote'].get('scoring_method', 'id'))
        if cfg.get('post_bbox_score_thresh', None) is not None:
            keep = _boxes[:, 4] > cfg['post_bbox_score_thresh'][cls - 1]
            _boxes = _boxes[keep]
        ix = _boxes.new_full((_boxes.shape[0], 1), b_ix)
        c = _boxes.new_full((_boxes.shape[0], 1), cls)
        post_bboxes = torch.cat([ix, _boxes, c], dim=1)
        detected_bboxes.append(post_bboxes)

    if len(detected_bboxes) == 0:
        return loc_pred.new_zeros((1, 7))
    else:
        detected_bboxes = torch.cat(detected_bboxes, dim=0)
        bboxes = []
        for b_ix in range(B):
            # keep the top_n well-predicted bboxes per image
            _boxes = detected_bboxes[detected_bboxes[:, 0] == b_ix]
            top_n = cfg.get('top_n', -1)
            if top_n > 0 and top_n < _boxes.shape[0]:
                _, order = _boxes[:, 5].sort(descending=True)
                _boxes = _boxes[order[:top_n]]
            bboxes.append(_boxes)
        return torch.cat(bboxes, dim=0)
