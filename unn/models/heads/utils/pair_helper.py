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
        for c_ix in range(1, C + 1):
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

def pair_box_transform(bbox1, bbox2):

    delta_x = float(bbox2[0] - bbox1[0]) / float(bbox1[2] - bbox1[0])
    delta_y = float(bbox2[1] - bbox1[1]) / float(bbox1[3] - bbox1[1])

    delta_h = float(bbox2[2] - bbox2[0]) / float(bbox1[2] - bbox1[0])
    delta_w = float(bbox2[3] - bbox2[1]) / float(bbox1[3] - bbox1[1])

    return np.array([delta_x, delta_y, delta_h, delta_w], dtype='float32')

def pair_pos_score(bbox1, bbox2, pred_pos):

    N = pred_pos.shape[0]
    delta_x = (bbox2[:, 0] - bbox1[:, 0]).astype(np.float32) / (bbox1[:, 2] - bbox1[:, 0]).astype(np.float32)
    delta_y = (bbox2[:, 1] - bbox1[:, 1]).astype(np.float32) / (bbox1[:, 3] - bbox1[:, 1]).astype(np.float32)

    delta_h = (bbox2[:, 2] - bbox2[:, 0]).astype(np.float32) / (bbox1[:, 2] - bbox1[:, 0]).astype(np.float32)
    delta_w = (bbox2[:, 3] - bbox2[:, 1]).astype(np.float32) / (bbox1[:, 3] - bbox1[:, 1]).astype(np.float32)
    delta_x = delta_x.reshape((N,1))
    delta_y = delta_y.reshape((N,1))
    delta_h = delta_h.reshape((N,1))
    delta_w = delta_w.reshape((N,1))
    object_pos = np.concatenate([delta_x, delta_y, delta_h, delta_w], axis=1)
    delta = pred_pos - object_pos
    scores = np.sum((delta * delta) / 0.18, axis=1)
    scores = np.exp(-scores)

    return scores
