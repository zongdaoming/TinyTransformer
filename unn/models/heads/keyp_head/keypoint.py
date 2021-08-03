import logging
import pdb
import torch

from ..utils.bbox_helper import clip_bbox
from ..utils.bbox_helper import filter_by_size
from ..utils.box_sampler import sample
from ..utils.matcher import match

logger = logging.getLogger('global')


def generate_softmax_labels(rois, keyps, label_h, label_w):
    """
    Arguments:
        rois (FloatTensor): [R, 4]
        keyps (FloatTensor): [R, K, 3], (x, y, v)
        label_w, label_h (int): label size

    Returns:
        labels (LongTensor): [R, K]
    """
    assert (rois.shape[0] == keyps.shape[0])
    R, K = keyps.shape[:2]
    scale_x = label_w / torch.clamp(rois[:, 2] - rois[:, 0] + 1, min=1)
    scale_y = label_h / torch.clamp(rois[:, 3] - rois[:, 1] + 1, min=1)
    keyp_x = (keyps[:, :, 0] - rois[:, 0, None] + 0.5) * scale_x[:, None]
    keyp_y = (keyps[:, :, 1] - rois[:, 1, None] + 0.5) * scale_y[:, None]
    keyp_x = keyp_x.to(dtype=torch.int64)
    keyp_y = keyp_y.to(dtype=torch.int64)
    keyp_v = keyps[:, :, 2] > 0

    mask = ((keyp_x >= 0) & (keyp_x < label_w) & (keyp_y >= 0) & (keyp_y < label_h) & keyp_v)
    labels = -torch.ones((R, K), device=rois.device, dtype=torch.int64)
    labels[mask] = keyp_y[mask] * label_w + keyp_x[mask]
    return labels


def keypoint_targets(sample_record, proposals, gt_bboxes, gt_keyps, image_info, cfg):
    """
    Arguments:
        sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)], sampling results from bbox head
        proposals (FloatTensor, fp32): [B, >=5] (batch_ix, x1, y1, x2, y2, ...)
        gt_bboxes (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
        gt_keyps (list of FloatTensor): [B, num_gts, K, 3], (x, y, flag)
        image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

    Returns:
        rois (FloatTensor): [R, 5] (b_ix, x1,y1,x2,y2)
        keyps_target (LongTensor): [B, K]
    """
    B = len(gt_keyps)
    K = gt_keyps[0].shape[1]

    batch_rois = []
    batch_keyps_target = []
    for b_ix in range(B):
        rois = proposals[proposals[:, 0] == b_ix]
        if rois.shape[0] == 0: continue
        rois = rois[:, 1:1 + 4]

        # filter bboxes and keyps which are too small
        _gt_bboxes, filter_inds = filter_by_size(gt_bboxes[b_ix], min_size=1)
        if _gt_bboxes.numel() == 0: continue
        filter_inds = filter_inds.nonzero().squeeze(1).cpu().numpy()
        _gt_keyps = gt_keyps[b_ix][filter_inds]

        # resample or not, if use ohem loss, supposed to be True
        # although with a litte difference, resample or not almost has no effect here
        if cfg["resample"]:
            keep_inds = _gt_keyps[:, :, 2].max(dim=1)[0] > 0
            _gt_bboxes = _gt_bboxes[keep_inds]
            _gt_keyps = _gt_keyps[keep_inds]

            if _gt_bboxes.shape[0] == 0: continue
            rois = torch.cat([rois, _gt_bboxes[:, :4]], dim=0)
            rois = clip_bbox(rois.floor(), image_info[b_ix])

            rois_target_gt = match(rois, _gt_bboxes, cfg['matcher'])
            pos_inds, _ = sample(rois_target_gt, cfg['sampler'])
            pos_target_gt = rois_target_gt[pos_inds]
        else:
            if not sample_record[b_ix]: continue
            pos_inds, pos_target_gt = sample_record[b_ix]
            rois = torch.cat([rois, _gt_bboxes[:, :4]], dim=0)
            rois = clip_bbox(rois.floor(), image_info[b_ix])
        if pos_inds.numel() == 0: continue

        # acquire target keyps for sampled rois
        pos_rois = rois[pos_inds]
        pos_keyps_target = generate_softmax_labels(pos_rois, _gt_keyps[pos_target_gt], cfg['label_h'], cfg['label_w'])

        ix = pos_rois.new_full((pos_rois.shape[0], 1), b_ix)
        pos_rois = torch.cat([ix, pos_rois], dim=1)
        batch_rois.append(pos_rois)
        batch_keyps_target.append(pos_keyps_target)

    if len(batch_rois) == 0:
        logger.warning('no positive rois found for keypoint')
        rois = rois.new_zeros((1, 5))
        keyps_target = rois.new_full((1, K), -1, dtype=torch.int64)
    else:
        rois = torch.cat(batch_rois, dim=0)
        keyps_target = torch.cat(batch_keyps_target, dim=0)

    return rois, keyps_target


def predict_keypoints(rois, heatmap):
    """
    Arguments:
        rois (FloatTensor, fp32): [R, >=5] (batch_ix, x1, y1, x2, y2, ...)
        heatmap (FloatTensor, fp32): [R, K, h, w]

    Returns:
        keyps (FloatTensor): [R, K, 3] (x, y, score)
    """
    rois_tl_x = rois[:, 1]
    rois_tl_y = rois[:, 2]
    rois_w = torch.clamp(rois[:, 3] - rois[:, 1] + 1, min=1)
    rois_h = torch.clamp(rois[:, 4] - rois[:, 2] + 1, min=1)

    R, K, H, W = heatmap.shape
    heatmap = heatmap.reshape(R, K, -1)
    score, index = heatmap.max(dim=2)
    x = (index % W).float()
    y = (index // W).float()
    x = (x + 0.5) / W * rois_w[:, None] + rois_tl_x[:, None]
    y = (y + 0.5) / H * rois_h[:, None] + rois_tl_y[:, None]
    return torch.stack([x, y, score], dim=2)
