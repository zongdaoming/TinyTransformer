import time
import logging
import torch
import numpy as np

from ....extensions import gpu_iou_overlap

GPU_MEMORY = None

logger = logging.getLogger('global')
def allow_empty_tensor(num=1):
    """Return an empty tensor directly if any of first `num` argument is empty"""

    def decorate(func):
        def wrapper(*args, **kwargs):
            for arg in args[:num]:
                if torch.is_tensor(arg) and arg.numel() == 0:
                    return arg.new_zeros((0, ))
            return func(*args, **kwargs)

        return wrapper

    return decorate


@allow_empty_tensor(1)
def filter_by_size(boxes, min_size, start_index=0):
    s = start_index
    w = boxes[:, s + 2] - boxes[:, s + 0] + 1
    h = boxes[:, s + 3] - boxes[:, s + 1] + 1
    filter_inds = (w > min_size) & (h > min_size)
    return boxes[filter_inds], filter_inds


@allow_empty_tensor(2)
def bbox_iou_overlaps(b1, b2):
    if not b1.is_cuda:
        return vanilla_bbox_iou_overlaps(b1, b2)

    global GPU_MEMORY
    gbytes = 1024.0**3
    if GPU_MEMORY is None:
        GPU_MEMORY = torch.cuda.get_device_properties(b1.device.index).total_memory
    alloated_memory = torch.cuda.memory_allocated()
    spare_memory = 0.5 * gbytes
    available_memory = GPU_MEMORY - alloated_memory - spare_memory
    size = b1.shape[0] * b2.shape[0]
    needed_memory = 2 * size * 4

    if needed_memory < available_memory:
        ious = gpu_iou_overlap(b1, b2)
    else:
        ious = vanilla_bbox_iou_overlaps(b1.cpu(), b2.cpu())
        res_memory = size * 4
        if res_memory < available_memory:
            ious = ious.to(b1.device)
    return ious


@allow_empty_tensor(2)
def vanilla_bbox_iou_overlaps(b1, b2):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise.
    """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    area2 = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:4], b2[:, 2:4])
    wh = (rb - lt + 1).clamp(min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    union_area = area1[:, None] + area2 - inter_area
    return inter_area / torch.clamp(union_area, min=1)


@allow_empty_tensor(2)
def bbox_iof_overlaps(b1, b2):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-former-box pair-wise
    """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:4], b2[:, 2:4])
    wh = (rb - lt + 1).clamp(min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    return inter_area / torch.clamp(area1[:, None], min=1)


@allow_empty_tensor(1)
def xywh2xyxy(boxes, stacked=False):
    """(x, y, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w - 1
    ymax = cy + 0.5 * h - 1
    if stacked:
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        return xmin, ymin, xmax, ymax


@allow_empty_tensor(1)
def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return torch.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h


@allow_empty_tensor(2)
def bbox2offset(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    assert boxes.shape[0] == gt_boxes.shape[0]
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * torch.log(gt_widths / ex_widths)
    offset_dh = wh * torch.log(gt_heights / ex_heights)
    offset = torch.stack((offset_dx, offset_dy, offset_dw, offset_dh), dim=1)
    return offset


@allow_empty_tensor(2)
def offset2bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = torch.clamp(dw, max=np.log(1000. / 16.))
    dh = torch.clamp(dh, max=np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = offset.new_zeros(offset.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


@allow_empty_tensor(2)
def offset2tiled_bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return boxes.new_zeros((1, offset.shape[1]))
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = torch.clamp(dw, max=np.log(1000. / 16.))
    dh = torch.clamp(dh, max=np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = offset.new_zeros(offset.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


@allow_empty_tensor(1)
def normalize_offset(offset, mean, std):
    mean = offset.new_tensor(mean).reshape(-1, 4)
    std = offset.new_tensor(std).reshape(-1, 4)
    return (offset - mean) / std


@allow_empty_tensor(1)
def unnormalize_offset(offset, mean, std):
    mean = offset.new_tensor(mean).reshape(-1, 4)
    std = offset.new_tensor(std).reshape(-1, 4)
    return offset * std + mean


@allow_empty_tensor(1)
def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = torch.clamp(bbox[:, 0], min=0, max=w - 1)
    bbox[:, 1] = torch.clamp(bbox[:, 1], min=0, max=h - 1)
    bbox[:, 2] = torch.clamp(bbox[:, 2], min=0, max=w - 1)
    bbox[:, 3] = torch.clamp(bbox[:, 3], min=0, max=h - 1)
    return bbox


@allow_empty_tensor(1)
def clip_tiled_boxes(bbox, img_size):
    assert bbox.shape[1] % 4 == 0, \
        'bbox.shape[1] is {}, but must be divisible by 4'.format(bbox.shape[1])
    h, w = img_size[:2]
    bbox[:, 0::4] = torch.clamp(bbox[:, 0::4], min=0, max=w - 1)
    bbox[:, 1::4] = torch.clamp(bbox[:, 1::4], min=0, max=h - 1)
    bbox[:, 2::4] = torch.clamp(bbox[:, 2::4], min=0, max=w - 1)
    bbox[:, 3::4] = torch.clamp(bbox[:, 3::4], min=0, max=h - 1)
    return bbox


@allow_empty_tensor(1)
def flip_tiled_bboxes(boxes, width):
    boxes_flipped = boxes.clone()
    boxes_flipped[:, 0::4] = width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = width - boxes[:, 0::4] - 1
    return boxes_flipped


@allow_empty_tensor(2)
def box_voting(top_dets, all_dets, thresh, scoring_method='id', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.cpu().numpy().copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_iou_overlaps(top_boxes, all_boxes)
    top_to_all_overlaps = top_to_all_overlaps.cpu().numpy()
    all_boxes = all_boxes.cpu().numpy()
    all_scores = all_scores.cpu().numpy()
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'id':
            # Identity, nothing to do
            pass
        elif scoring_method == 'temp_avg':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'avg':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'iou_avg':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'generalized_avg':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'quasi_sum':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError('Unknown scoring method {}'.format(scoring_method))
    top_dets_out = torch.from_numpy(top_dets_out).to(top_dets)
    return top_dets_out


def test_bbox_iou_overlaps():
    b1 = torch.FloatTensor([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5]])
    b2 = torch.FloatTensor([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5], [100, 100, 200, 200]])
    overlaps = bbox_iou_overlaps(b1, b2)
    print(overlaps)


def test_bbox_iof_overlaps():
    b1 = torch.FloatTensor([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5]])
    b2 = torch.FloatTensor([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5], [100, 100, 200, 200]])
    overlaps = bbox_iof_overlaps(b1, b2)
    print(overlaps)


def test_xyxy_xywh():
    b1 = torch.FloatTensor([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5]])
    b2 = xyxy2xywh(b1)
    b2 = torch.stack(b2, dim=1)
    b3 = xywh2xyxy(b2)
    b3 = torch.stack(b3, dim=1)
    print(b1)
    print(b2)
    print(b3)


def test_offset():
    b1 = torch.FloatTensor([[0, 0, 4, 4], [1, 2, 3, 5], [4, 4, 5, 5]])
    tg = torch.FloatTensor([[1, 1, 5, 5], [0, 2, 4, 5], [4, 4, 5, 5]])
    offset = bbox2offset(b1, tg)
    print(offset)
    pred = offset2bbox(b1, offset)
    print(pred)


def test_clip_bbox():
    b1 = torch.FloatTensor([[0, 0, 9, 29], [1, 2, 19, 39], [4, 4, 59, 59]])
    print(b1)
    b2 = clip_bbox(b1, (30, 35))
    print(b2)


def test_box_voting():
    b1 = torch.FloatTensor([[0, 0, 4, 4, 0.6], [5, 5, 5, 5, 0.9]])
    b2 = torch.FloatTensor([[0, 0, 4, 4, 0.1], [0, 0, 3, 5, 0.9], [5, 5, 5, 5, 0.9], [100, 100, 200, 200, 10]])
    voted_box = box_voting(b1, b2, 0.5, 'id')
    print(voted_box)


def test_iou(iou_fn, a, b):
    n = 5
    s = time.time()
    for i in range(n):
        iou = iou_fn(a, b)
    del iou
    torch.cuda.synchronize()
    e = time.time()
    t = e - s
    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024
    return t, memory


def rand_bbox(n):
    box = torch.randn(n, 4).cuda() * 10
    x1 = torch.min(box[:, 0], box[:, 2])
    x2 = torch.max(box[:, 0], box[:, 2])
    y1 = torch.min(box[:, 1], box[:, 3])
    y2 = torch.max(box[:, 1], box[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1)


if __name__ == '__main__':
    # test_bbox_iou_overlaps()
    # test_bbox_iof_overlaps()
    # test_xyxy_xywh()
    # test_offset()
    # test_clip_bbox()
    test_box_voting()
