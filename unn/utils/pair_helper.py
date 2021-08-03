import torch
import torch.nn
import numpy as np


def cal_iou(b1, b2):

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[0], b2[:, 0].reshape(1,-1))
    inter_ymin = np.maximum(b1[1], b2[:, 1].reshape(1,-1))
    inter_xmax = np.minimum(b1[2], b2[:, 2].reshape(1,-1))
    inter_ymax = np.minimum(b1[3], b2[:, 3].reshape(1,-1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    union_area1 = area1 + area2
    union_area2 = union_area1 - inter_area
    return inter_area / np.maximum(union_area2, 1)



def pair_nms(pair_rois, cfg):

    B = int(np.max(pair_rois, axis=0)[0])
    C = int(np.max(pair_rois, axis=0)[11])
    keep_pair = []
    for b_ix in range(B + 1):
        batch_ix = np.where(pair_rois[:, 0] == b_ix)[0]
        batch_pair = pair_rois[batch_ix]
        for c_ix in range(C + 1):
            cls_ix = np.where(batch_pair[:, 11] == c_ix)
            cls_pair = batch_pair[cls_ix]
            scores = cls_pair[:,12]
            idx = np.argsort(scores)[::-1]
            while len(idx) > 0:
                max_idx = idx[0]
                keep_pair.append(cls_pair[max_idx])
                if len(idx) == 1:
                    break
                iou1 = cal_iou(cls_pair[max_idx][1:5], cls_pair[idx[1:], 1:5])
                iou2 = cal_iou(cls_pair[max_idx][6:10], cls_pair[idx[1:], 6:10])
                ids = np.where((((iou1 > cfg['iou_thresh_hi'])|(iou1 < cfg['iou_thresh_lo']))&((iou2 > cfg['iou_thresh_hi']) | (iou2 < cfg['iou_thresh_lo']))))[1]
                idx = idx[ids + 1]
    return np.array(keep_pair)
