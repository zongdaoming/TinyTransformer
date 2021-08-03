#coding: utf-8
from ..utils import bbox_helper
from ..utils.pair_helper import pair_nms
from ..utils.pair_helper import pair_box_transform
from ..utils.pair_helper import pair_pos_score
import pdb
import ctypes

import math
import numpy as np
import torch
from torch.autograd import Variable
import logging
import json
logger = logging.getLogger('global')
history = [0, 0]

def to_np_array(x):
    if x is None:
        return x
    if isinstance(x, Variable): x = x.data
    return x.cpu().float().numpy() if torch.is_tensor(x) else x

def compute_proposal_targets2(all_rois, pair_rois, num_classes, cfg, gt_bboxes, gt_assos, roiTable, image_info, predict_type='rcnn', ignore_regions=None):
    B = len(image_info)
    C = num_classes
    N = pair_rois.shape[0]  # 267，前一步makepair生成了 267 个pair
    gt_cls = np.array([0 for _ in range(N)]).astype(np.int32)
    all_rois, gt_assos, image_info = map(to_np_array, [all_rois, gt_assos, image_info])
    pos_ix = []
    neg_ix = []
    gt_cls = [0 for __ in range(N)]
    gt_binary_cls = [[0 for __ in range(C)] for _ in range(N)]
    for b_ix in range(B):
        idx = np.where(all_rois[:,0] == b_ix)[0]
        rois = all_rois[idx, 1: 5]  # (100, 4) 该图片下的检测bbox
        gts = gt_bboxes[b_ix]  # (16, 5)
        gts = to_np_array(gts) # (1, 13) 该 图片 下的 gt 关联
        gtasso = gt_assos[b_ix].cpu().numpy()  # [idx1, idx2, item['label']] + item['bbox1'] + [item['label1']] + item['bbox2'] + [item['label2']]
        dt_pairs = pair_rois[pair_rois[:, 0] == b_ix]
        R = rois.shape[0]  
        G = gts.shape[0]
        if R == 0:
            continue
        if G > 0:
            overlaps = bbox_helper.bbox_iou_overlaps(torch.from_numpy(rois), torch.from_numpy(gts)).numpy() # (100, 16)
            argmax_overlaps = overlaps.argmax(axis=1)  # 每一个检测框与gt iou重叠最大的 bbox 的索引 (100,)
            max_overlaps = overlaps.max(axis=1)  # 每一个检测框与gt iou重叠最大的值 (100, )
            if cfg.get('allow_low_quality_match', False):
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                gt_max_overlaps = overlaps.max(axis=0)
                gt_pos_g_ix = np.where(gt_max_overlaps > cfg['negative_iou_thresh'])[0]
                gt_pos_r_ix = gt_argmax_overlaps[gt_pos_g_ix]
                gt_pos_g_ix, return_index = np.unique(gt_pos_g_ix, return_index=True)
                gt_pos_r_ix = gt_pos_r_ix[return_index]
                max_overlaps[gt_pos_r_ix] = 1.0

            pos_r_ix = np.where(max_overlaps > cfg['positive_iou_thresh'])[0] # (91, ) 选择该图下的检测的正例的索引
            pos_g_ix = argmax_overlaps[pos_r_ix]  # (91, ) 改图下的检测的正例的匹配到的gt的索引
            pos_r_ix, return_index = np.unique(pos_r_ix, return_index=True)
            pos_g_ix = pos_g_ix[return_index]
            GG = gtasso.shape[0]
            #for m in range(dt_pairs.shape[0]):
            #    x = dt_pairs[m, 1]
            #    y = dt_pairs[m, 2]
            #    if x in pos_r_ix and y in pos_r_ix:
            #        continue
            #    else:
            #        id = roiTable[(x,y)]
            #        neg_ix.append(id)
            #        gt_cls[id] = 0
            
            for k in range(GG):  # 对于每一个 gt
                if gtasso[k][2] == 0:  # 如果是背景，则略过
                    continue
                x = gtasso[k][0]  # pair id 1
                y = gtasso[k][1]  # pair id 2
                for i in range(len(pos_r_ix)):
                    for j in range(len(pos_r_ix)):
                        if x == pos_g_ix[i] and y == pos_g_ix[j]:  # 只有都匹配到了，才算 pos
                            rx = idx[pos_r_ix[i]]
                            ry = idx[pos_r_ix[j]]
                            if not (rx, ry) in roiTable:
                                continue
                            assert((rx,ry) in roiTable)
                            id = roiTable[(rx,ry)]
                            pos_ix.append(id)
                            if predict_type == 'rcnn':
                                gt_cls[id] = gtasso[k][2]
                                gt_binary_cls[id][gtasso[k][2].astype(np.int32)] = 1
                            else:
                                gt_cls[id] = 1
                                gt_binary_cls[id][0] = 1
    
    neg_ix = np.array(list(set([_ for _ in range(N)]) - set(pos_ix))) # 实际本batch中得到的负例的个数，237
    batch_size = cfg['batch_size']
    all_num = batch_size * B  # 128 * 4 = 512
    #pos_num = int(all_num * cfg['positive_percent'])  # 128
    pos_ix = np.array(pos_ix)  # 实际本batch中得到的正例的个数，根绝某规则选择出的正例(iou。。。。) (30, )
    #if pos_ix.shape[0] > pos_num:  # 如果实际的正例比我们想要的正例数目还要多，则进行随机算则
    #    keep_ix = np.random.choice(len(pos_ix), size = pos_num, replace = True)  # 随机选择128个正例，降采样
    #    pos_ix = pos_ix[keep_ix]
    #    if cfg.get('dynamic_batch_size', False):  # 动态batch size
    #        all_num = int(len(pos_ix) / cfg['positive_percent'])  # 增大 batch size，并不一定是 512 了

    #neg_num = all_num - len(pos_ix)  # 我们想要的负例的个数 (482, )
    #if len(neg_ix) == 0:
    #    neg_ix = np.array([0])
    #keep_ix = np.random.choice(len(neg_ix), size = neg_num, replace = True)
    #neg_ix = neg_ix[keep_ix]  # (482, )
    neg_cls = [0 for __ in range(len(neg_ix))]
    neg_binary_cls = [[0 for __ in range(C)] for _ in range(len(neg_ix))]
    gt_cls = np.array(gt_cls)
    gt_cls = gt_cls[pos_ix.astype(np.int32)]
    gt_binary_cls = np.array(gt_binary_cls)
    gt_binary_cls = gt_binary_cls[pos_ix.astype(np.int32)]
    neg_ix = np.array(neg_ix)
    neg_cls = np.array(neg_cls)
    neg_binary_cls = np.array(neg_binary_cls)

    if len(pos_ix) == 0:
        dt = neg_ix
        gt_cls = neg_cls
        gt_binary_cls = neg_binary_cls
    else:
        dt = np.hstack([pos_ix, neg_ix])  # 正负例的索引 stack 起来
        gt_cls = np.hstack([gt_cls, neg_cls])
        gt_binary_cls = np.vstack([gt_binary_cls, neg_binary_cls])
    
    pdb.set_trace()
    if cfg.get('cls_type', 'softmax') == 'softmax':
        return dt, gt_cls, gt_cls.shape[0]
    else:
        return dt, gt_binary_cls, gt_binary_cls.shape[0]

def compute_proposal_targets_gt(all_rois, pair_rois, num_classes, cfg, gt_bboxes, gt_assos, roiTable, image_info, predict_type='rcnn', ignore_regions=None):
    '''
    gt_assos: [N, 3] rois1_ix, rois2_ix, cls
    '''
    #pdb.set_trace()
    B = len(image_info)
    C = num_classes
    N = pair_rois.shape[0]
    gt_cls = np.array([0 for _ in range(N)]).astype(np.int32)
    all_rois, gt_assos, image_info = map(to_np_array, [all_rois, gt_assos, image_info])
    pos_ix = []
    gt_cls = [0 for __ in range(N)]
    gt_binary_cls = [[0 for __ in range(C)] for _ in range(N)]
    for b_ix in range(B):
        idx = np.where(all_rois[:,0] == b_ix)[0]
        rois = all_rois[idx, 1: 5]
        gts = gt_bboxes[b_ix]
        gts = to_np_array(gts)
        gtasso = gt_assos[b_ix].cpu().numpy()
        R = rois.shape[0]
        G = gtasso.shape[0]
        if R == 0:
            continue
        if G > 0:
            overlaps = bbox_helper.bbox_iou_overlaps(torch.from_numpy(rois), torch.from_numpy(gts)).numpy()
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            if cfg.get('allow_low_quality_match', False):
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                gt_max_overlaps = overlaps.max(axis=0)
                gt_pos_g_ix = np.where(gt_max_overlaps > cfg['negative_iou_thresh'])[0]
                gt_pos_r_ix = gt_argmax_overlaps[gt_pos_g_ix]
                gt_pos_g_ix, return_index = np.unique(gt_pos_g_ix, return_index=True)
                gt_pos_r_ix = gt_pos_r_ix[return_index]
                max_overlaps[gt_pos_r_ix] = 1.0

            pos_r_ix = np.where(max_overlaps > cfg['positive_iou_thresh'])[0]
            pos_g_ix = argmax_overlaps[pos_r_ix]
            pos_r_ix, return_index = np.unique(pos_r_ix, return_index=True)
            pos_g_ix = pos_g_ix[return_index]
            GG = gtasso.shape[0]
            added_gt = set()
            for k in range(GG):
                if gtasso[k][2] == 0:
                    continue
                x = gtasso[k][0]
                y = gtasso[k][1]
                for i in range(len(pos_r_ix)):
                    for j in range(len(pos_r_ix)):
                        if x == pos_g_ix[i] and y == pos_g_ix[j]:
                            rx = idx[pos_r_ix[i]]
                            ry = idx[pos_r_ix[j]]
                            if not (rx, ry) in roiTable:
                                continue
                            assert((rx,ry) in roiTable)
                            id = roiTable[(rx,ry)]
                            pos_ix.append(id)
                            added_gt.add(k)
                            if predict_type == 'rcnn':
                                gt_cls[id] = gtasso[k][2]
                                gt_binary_cls[id][gtasso[k][2].astype(np.int32)] = 1
                            else:
                                gt_cls[id] = 1
                                gt_binary_cls[id][0] = 1
            overlaps = overlaps.T
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)

            no_added_gt = set(list(range(GG))) - added_gt
            if len(no_added_gt) <= 1:
                continue
            for k in no_added_gt: # for not added gt
                #pdb.set_trace()
                x = int(gtasso[k][0])
                y = int (gtasso[k][1])
                if max_overlaps[x] == 0 or max_overlaps[y] == 0:
                    continue
                rx = argmax_overlaps[x]
                ry = argmax_overlaps[y]
                rx = idx[rx]
                ry = idx[ry]
                if not (rx, ry) in roiTable:
                    continue
                assert((rx,ry) in roiTable)
                id = roiTable[(rx,ry)]
                pos_ix.append(id)
                if predict_type == 'rcnn':
                    gt_cls[id] = gtasso[k][2] # todo 类别没有对应好，如果要对应类别，就不能去max
                    gt_binary_cls[id][gtasso[k][2].astype(np.int32)] = 1
                else:
                    gt_cls[id] = 1
                    gt_binary_cls[id][0] = 1

    neg_ix = np.array(list(set([_ for _ in range(N)]) - set(pos_ix)))
    #pdb.set_trace()
    batch_size = cfg['batch_size']
    all_num = batch_size * B
    pos_num = int(all_num * cfg['positive_percent'])
    pos_ix = np.array(pos_ix)
    if pos_ix.shape[0] > pos_num:
        keep_ix = np.random.choice(len(pos_ix), size = pos_num, replace = True)
        pos_ix = pos_ix[keep_ix]
        if cfg.get('dynamic_batch_size', False):
            all_num = int(len(pos_ix) / cfg['positive_percent'])

    neg_num = all_num - len(pos_ix)
    if len(neg_ix) == 0:
        neg_ix = np.array([0])
    keep_ix = np.random.choice(len(neg_ix), size = neg_num, replace = True)
    neg_ix = neg_ix[keep_ix]
    neg_cls = [0 for __ in range(len(keep_ix))]
    neg_binary_cls = [[0 for __ in range(C)] for _ in range(len(keep_ix))]
    gt_cls = np.array(gt_cls)
    gt_cls = gt_cls[pos_ix.astype(np.int32)]
    gt_binary_cls = np.array(gt_binary_cls)
    gt_binary_cls = gt_binary_cls[pos_ix.astype(np.int32)]
    neg_ix = np.array(neg_ix)
    neg_cls = np.array(neg_cls)
    neg_binary_cls = np.array(neg_binary_cls)

    if len(pos_ix) == 0:
        dt = neg_ix
        gt_cls = neg_cls
        gt_binary_cls = neg_binary_cls
    else:
        dt = np.hstack([pos_ix, neg_ix])
        gt_cls = np.hstack([gt_cls, neg_cls])
        gt_binary_cls = np.vstack([gt_binary_cls, neg_binary_cls])
    idx = np.arange(len(dt))
    np.random.shuffle(idx)
    dt = dt[idx]
    gt_cls = gt_cls[idx]
    gt_binary_cls = gt_binary_cls[idx]
    #pdb.set_trace()
    if cfg.get('cls_type', 'softmax') == 'softmax':
        return dt, gt_cls, gt_cls.shape[0]
    else:
        return dt, gt_binary_cls, gt_binary_cls.shape[0]

def compute_proposal_targets(all_rois, pair_rois, num_classes, cfg, gt_bboxes, gt_assos, roiTable, image_info, predict_type='rcnn', ignore_regions=None):
    '''
    gt_assos: [N, 3] rois1_ix, rois2_ix, cls
    '''
    #pdb.set_trace()
    B = len(image_info)
    C = num_classes
    N = pair_rois.shape[0]
    gt_cls = np.array([0 for _ in range(N)]).astype(np.int32)
    all_rois, gt_assos, image_info = map(to_np_array, [all_rois, gt_assos, image_info])
    pos_ix = []
    gt_cls = [0 for __ in range(N)]
    gt_binary_cls = [[0 for __ in range(C)] for _ in range(N)]
    for b_ix in range(B):
        idx = np.where(all_rois[:,0] == b_ix)[0]
        rois = all_rois[idx, 1: 5]
        gts = gt_bboxes[b_ix]
        gts = to_np_array(gts)
        gtasso = gt_assos[b_ix].cpu().numpy()
        R = rois.shape[0]
        G = gts.shape[0]
        if R == 0:
            continue
        if G > 0:
            overlaps = bbox_helper.bbox_iou_overlaps(torch.from_numpy(rois), torch.from_numpy(gts)).numpy()
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            if cfg.get('allow_low_quality_match', False):
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                gt_max_overlaps = overlaps.max(axis=0)
                gt_pos_g_ix = np.where(gt_max_overlaps > cfg['negative_iou_thresh'])[0]
                gt_pos_r_ix = gt_argmax_overlaps[gt_pos_g_ix]
                gt_pos_g_ix, return_index = np.unique(gt_pos_g_ix, return_index=True)
                gt_pos_r_ix = gt_pos_r_ix[return_index]
                max_overlaps[gt_pos_r_ix] = 1.0

            pos_r_ix = np.where(max_overlaps > cfg['positive_iou_thresh'])[0]
            pos_g_ix = argmax_overlaps[pos_r_ix]
            pos_r_ix, return_index = np.unique(pos_r_ix, return_index=True)
            pos_g_ix = pos_g_ix[return_index]
            GG = gtasso.shape[0]
            for k in range(GG):
                if gtasso[k][2] == 0:
                    continue
                x = gtasso[k][0]
                y = gtasso[k][1]
                for i in range(len(pos_r_ix)):
                    for j in range(len(pos_r_ix)):
                        if x == pos_g_ix[i] and y == pos_g_ix[j]:
                            rx = idx[pos_r_ix[i]]
                            ry = idx[pos_r_ix[j]]
                            if not (rx, ry) in roiTable:
                                continue
                            assert((rx,ry) in roiTable)
                            id = roiTable[(rx,ry)]
                            pos_ix.append(id)
                            if predict_type == 'rcnn':
                                gt_cls[id] = 1
                                gt_binary_cls[id][gtasso[k][2].astype(np.int32)] = 1
                            else:
                                gt_cls[id] = 1
                                gt_binary_cls[id][0] = 1
    neg_ix = np.array(list(set([_ for _ in range(N)]) - set(pos_ix)))
    neg_cls = [0 for __ in range(len(neg_ix))]
    neg_binary_cls = [[0 for __ in range(C)] for _ in range(len(neg_ix))]
    gt_cls = np.array(gt_cls)
    pos_ix = np.array(pos_ix)
    gt_cls = gt_cls[pos_ix.astype(np.int32)]
    gt_binary_cls = np.array(gt_binary_cls)
    gt_binary_cls = gt_binary_cls[pos_ix.astype(np.int32)]
    neg_ix = np.array(neg_ix)
    neg_cls = np.array(neg_cls)
    neg_binary_cls = np.array(neg_binary_cls)

    if len(pos_ix) == 0:
        dt = neg_ix
        gt_cls = neg_cls
        gt_binary_cls = neg_binary_cls
    else:
        dt = np.hstack([pos_ix, neg_ix])
        gt_cls = np.hstack([gt_cls, neg_cls])
        if len(neg_binary_cls) != 0:
            gt_binary_cls = np.vstack([gt_binary_cls, neg_binary_cls])
    idx = np.arange(len(dt))
    dt = dt[idx]
    gt_cls = gt_cls[idx]
    gt_binary_cls = gt_binary_cls[idx]
    if cfg.get('cls_type', 'softmax') == 'softmax':
        return dt, gt_cls, gt_cls.shape[0]
    else:
        return dt, gt_binary_cls, gt_binary_cls.shape[0]

def predict_assos(rois, pair_rois, pred_cls, image_info, cfg, tocaffe):
    '''
    :param cfg: config
    :param rois: [N, k] k>=7, batch_ix, x1, y1, x2, y2, score, cls
    :param pred_assos:[N, num_classes * 4, 1, 1]
    :param image_info:[N, 3]
    :return: assos: [M, 13], batch_ix, ax1, ay1, ax2, ay2, acls, bx1, by1, bx2, by2, bcls, cls, score
    '''
    rois, pair_rois, pred_cls = map(to_np_array, [rois, pair_rois, pred_cls])
    N, num_classes = pred_cls.shape[0], 3
    B = max(rois[:, 0].astype(np.int32))+1
    nmsed_assos = [np.zeros((1,13))]
    rois1 = rois[pair_rois[:, 1].astype(np.int32)]
    rois2 = rois[pair_rois[:, 2].astype(np.int32)]
    #pdb.set_trace()
    for cls in range(1, num_classes):
        #if cls == 1:
            # in body-face pair, the cls of object must be face
        #    idx = np.where(rois2[:,6] == 2)
        #else:
            # in body-hand pair, the cls of object must be hand
        #    idx = np.where(rois2[:,6] == 3)
        #idx = np.where(pred_cls[:, cls] == pred_cls.max(axis=0))
        #pdb.set_trace()
        idx = np.where(rois2[:, 6] > 0)
        tmp_pred_cls = pred_cls[idx]
        tmp_rois1 = rois1[idx]
        tmp_rois2 = rois2[idx]
        tmp_pair = pair_rois[idx]
        #scores = (tmp_pred_cls[:,cls] * tmp_rois1[:, 5] * tmp_rois2[:, 5]).reshape((-1,1))
        
        scores = tmp_pred_cls.reshape((-1,1))
        
        pair_cls = tmp_rois2[:, 6].copy()
        pair_cls[pair_cls == 2] = 1
        pair_cls[pair_cls == 3] = 2
        pair_cls = pair_cls.reshape((-1, 1))
       
        pair_cls = np.array([cls for _ in range(scores.shape[0])]).reshape((-1,1))
        batch = tmp_pair[:,0]
        batch = batch.reshape((-1,1))
        #nmsed_assos.append(np.hstack([batch, tmp_rois1[:,1:5], tmp_rois1[:,6].reshape((-1,1)), tmp_rois2[:, 1:5], tmp_rois2[:,6].reshape((-1,1)), pair_cls, scores]))
        batch_asso = np.hstack([batch, tmp_rois1[:,1:5], tmp_rois1[:,6].reshape((-1,1)), \
                tmp_rois2[:, 1:5], tmp_rois2[:,6].reshape((-1,1)), pair_cls, scores])
        nmsed_assos.append(batch_asso)
                

    if tocaffe:
        nmsed_assos = np.zeros((1,13))
    else:
        nmsed_assos = np.vstack(nmsed_assos)
    if cfg['top_n'] > 0:
        top_n_assos = [[0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0,0,0]]
        for b_ix in range(B):
            assos = nmsed_assos[nmsed_assos[:, 0] == b_ix]
            if assos.size == 0: continue
            scores = assos[:, -1]
            order = scores.argsort()[::-1][:cfg['top_n']]
            assos = assos[order]
            if cfg.get('score_thresh', None) is not None:
                keep = assos[:, -1] > cfg['score_thresh']
                #keep = np.bitwise_and(0.5 > assos[:, -1], assos[:, -1] > cfg['score_thresh'])
                assos = assos[keep]
            top_n_assos.append(assos)
        nmsed_assos = np.vstack(top_n_assos)
    return nmsed_assos


def predict_assos_2(rois, pair_rois, pred_cls, image_info, cfg, tocaffe):
    '''
    :param cfg: config
    :param rois: [N, k] k>=7, batch_ix, x1, y1, x2, y2, score, cls
    :param pred_assos:[N, num_classes * 4, 1, 1]
    :param image_info:[N, 3]
    :return: assos: [M, 13], batch_ix, ax1, ay1, ax2, ay2, acls, bx1, by1, bx2, by2, bcls, cls, score
    '''
    rois, pair_rois, pred_cls = map(to_np_array, [rois, pair_rois, pred_cls])
    N, num_classes = pred_cls.shape[0:2]
    B = max(rois[:, 0].astype(np.int32))+1
    nmsed_assos = [np.zeros((1,13))]
    #pdb.set_trace()
    rois1 = rois[pair_rois[:, 1].astype(np.int32)]
    rois2 = rois[pair_rois[:, 2].astype(np.int32)]
    for cls in range(1, num_classes):
        #if cls == 1:
            # in body-face pair, the cls of object must be face
        #    idx = np.where(rois2[:,6] == 2)
        #else:
            # in body-hand pair, the cls of object must be hand
        #    idx = np.where(rois2[:,6] == 3)
        idx = np.where(rois2[:, 6] > 0)
        tmp_pred_cls = pred_cls[idx]
        tmp_rois1 = rois1[idx]
        tmp_rois2 = rois2[idx]
        tmp_pair = pair_rois[idx]
        scores = (tmp_pred_cls[:,cls] * tmp_rois1[:, 5] * tmp_rois2[:, 5]).reshape((-1,1))
        #scores = tmp_pred_cls[:, cls].reshape((-1,1))
        # pair_cls = np.array([cls for _ in range(scores.shape[0])]).reshape((-1,1))
        pair_cls = tmp_rois2[:, 6].copy()
        pair_cls[pair_cls == 2] = 1
        pair_cls[pair_cls == 3] = 2
        pair_cls = pair_cls.reshape((-1, 1))
        batch = tmp_pair[:,0]
        batch = batch.reshape((-1,1))
        #nmsed_assos.append(np.hstack([batch, tmp_rois1[:,1:5], tmp_rois1[:,6].reshape((-1,1)), tmp_rois2[:, 1:5], tmp_rois2[:,6].reshape((-1,1)), pair_cls, scores]))
        batch_asso = np.hstack([batch, tmp_rois1[:,1:5], tmp_rois1[:,6].reshape((-1,1)), 
            tmp_rois2[:, 1:5], tmp_rois2[:,6].reshape((-1,1)), pair_cls, scores])
        if cfg.get('cls_top_n', 0) > 0:
            for b_ix in range(B):
                assos = batch_asso[batch_asso[:, 0] == b_ix]
                if assos.size == 0: continue
                scores = assos[:, -1]
                order = scores.argsort()[::-1][:cfg['cls_top_n']]
                assos = assos[order]
                if cfg.get('use_filter', None) is not None:
                    cls_triplet = cfg['asso_triplet'][cls - 1]
                    tmp = []
                    for line in assos:
                        acls = line[5].astype(int)
                        bcls = line[10].astype(int)
                        if (str(acls) in cls_triplet.keys() and bcls in cls_triplet[str(acls)]):
                           tmp.append(line)
                    if len(tmp) > 0:
                        assos = np.vstack(tmp)
                    else:
                        assos = []
                if len(assos) > 0:
                    nmsed_assos.append(assos)
        else:
            nmsed_assos.append(batch_asso)
                

    if tocaffe:
        nmsed_assos = np.zeros((1,13))
    else:
        nmsed_assos = np.vstack(nmsed_assos)
        if cfg.get('pair_nms', None):
            if cfg['pre_top_n'] > 0:
                top_n_assos = []
                for b_ix in range(B):
                    assos = nmsed_assos[nmsed_assos[:, 0] == b_ix]
                    if assos.size == 0: continue
                    scores = assos[:, -1]
                    order = scores.argsort()[::-1][:cfg['pre_top_n']]
                    assos = assos[order]
                    if cfg.get('pre_score_thresh', None) is not None:
                        keep = assos[:, -1] > cfg['pre_score_thresh']
                        assos = assos[keep]
                    top_n_assos.append(assos)
                nmsed_assos = np.vstack(top_n_assos)
            nmsed_assos = pair_nms(nmsed_assos, cfg['pair_nms'])
    if cfg['top_n'] > 0:
        top_n_assos = [[0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0,0,0]]
        for b_ix in range(B):
            assos = nmsed_assos[nmsed_assos[:, 0] == b_ix]
            if assos.size == 0: continue
            scores = assos[:, -1]
            order = scores.argsort()[::-1][:cfg['top_n']]
            assos = assos[order]
            if cfg.get('score_thresh', None) is not None:
                keep = assos[:, -1] > cfg['score_thresh']
                assos = assos[keep]
            top_n_assos.append(assos)
        nmsed_assos = np.vstack(top_n_assos)
    return nmsed_assos

