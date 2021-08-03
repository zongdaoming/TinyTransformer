# coding:utf-8
import functools
import logging
import time
import copy
import json
import math
import importlib
import pickle
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from .... import extensions as E
from ..utils.pair_helper import cal_iou
from ..utils.bbox_helper import clip_bbox
from ..utils import loss as L
from ..utils import accuracy as A
from ..utils import bbox_helper

from ..utils.assigner import map_rois_to_level
from ...initializer import initialize_from_cfg, init_weights_normal
from .pair import compute_proposal_targets, predict_assos, predict_assos_2, compute_proposal_targets_gt
from .matcher import SingleFramePostProcessorFactory
from .position import PositionTransform, PositionEmbedding
from .ibconv import build_ibconv
from unn.models.backbones.resnext_syncbn import ResNeXtBottleneck
from unn.models.attentions.siamese_attention import SiameseAttention
from unn.models.attentions.siamese_attention import SiameseAttentionPlus
from unn.models.functions.embedding import BoxPositionEmbedding
from unn.models.functions.embedding import ImagePositionEmbedding

import pdb

__all__ = ['GeneralAssociation', 'GeneralAssociation_union',] 

logger = logging.getLogger('global')

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().float().numpy() if torch.is_tensor(x) else x


def parse_bbox(file_path):
    all_bbox = {}
    for line in open(file_path):
        content = json.loads(line)
        if not "bbox" in content:
            continue
        image_id = content["image_id"]
        if image_id[-4] != ".":
            image_id = image_id + ".jpg"
        #content['score'] = 8.4 / (1 + math.exp(12.0 -content['score'] * 10.0))
        bbox = [0] + content["bbox"] + [content["score"]] + [content["label"]]
        if not image_id in all_bbox:
            all_bbox[image_id] = []
        all_bbox[image_id].append(bbox)
    for image_id in all_bbox.keys():
        all_bbox[image_id] = np.array(all_bbox[image_id])
    return all_bbox


class PairWiseNet(nn.Module):
    def __init__(self, inplanes, num_classes, cfg):
        super(PairWiseNet, self).__init__()
        self.origin_cfg = copy.deepcopy(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.tocaffe = self.cfg.get('tocaffe', False)
        self.cfg['num_classes'] = num_classes
        self.num_classes = num_classes
        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes
        self.roipool = E.build_generic_roipool(cfg['roipooling'])
        self.pool_size = cfg['roipooling']['pool_size']
        self.position_score = cfg.get('position_score', False)
        self.single_mdim = self.pool_size * self.pool_size * inplanes
        if self.cfg.get('pre_fc', None):
            self.pre_fc = nn.Linear(self.single_mdim, self.cfg['pre_fc'])
            self.single_mdim = self.cfg['pre_fc']
        else:
            self.pre_fc = None
        self.mdim = self.single_mdim * 2
        self.union_box = self.cfg.get('union_box', False)
        if not self.cfg.get('position', None) is None:
            # Only use position transform
            if self.cfg['position'] == 'naive':
                self.mdim += 14
            elif self.cfg['position'] == 'embedding':
            # Use position embedding, each element will be mapped to high dimension
                self.mdim += 14 * 256
            else:
                self.position_fc1 = nn.Linear(self.pool_size * self.pool_size * inplanes, 1024)
                self.position_relu = nn.ReLU(inplace=True)
                self.position_fc2 = nn.Linear(1024, 1024)
                self.position_fc3 = nn.Linear(1024, 14 * 256)
                self.mdim += 14 * 256
        self.predict_kernel = self.cfg.get('predict_kernel', None)
        if self.predict_kernel is not None:
            self.ibconv = build_ibconv(self.inplanes, self.pool_size, self.cfg)
            
            
        if self.cfg.get('similarity', False):
            # Use vector dot operation to assess the similarity
            self.mdim += self.single_mdim
        if self.union_box:
            self.mdim += self.single_mdim
    
        if self.cfg.get('use_filter', None):
            asso_triplet = []
            filename = cfg.get('use_filter')
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    mp = json.loads(line.strip())
                    asso_triplet.append(mp)
            self.origin_cfg['asso_triplet'] = asso_triplet
        if self.cfg.get('pre_filter', None) is not None:
            self.pre_filter = json.loads(open(self.cfg['pre_filter']).readlines()[0])
        self.cls_mdim = 0
        self.keep_origin_feature = False
        self.use_rpn = False
        self.output_predict = self.origin_cfg.get('output_predict', False)
        if self.cfg.get('binary_mask', None) is not None:
            self.binary_mask = json.loads(open(self.cfg['binary_mask']).readlines()[0])
            bbox_num = self.cfg['num_bbox_classes']
            self.binary_scale = [[0 for _ in range(self.num_classes)] for __ in range(bbox_num)]
            for k in self.binary_mask.keys():
                v = self.binary_mask[k]
                for item in v:
                    self.binary_scale[k][item] = 1
            self.binary_scale = torch.cuda.FloatTensor(self.binary_scale)
        else:
            self.binary_mask = None
        if self.cfg.get('element_wise_sum', False):
            self.mdim = self.single_mdim
        self.save_res = []

    def calIoU(self, b1, b2):
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
        inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
        inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
        inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
        inter_h = np.maximum(inter_xmax - inter_xmin, 0)
        inter_w = np.maximum(inter_ymax - inter_ymin, 0)
        inter_area = inter_h * inter_w
        union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
        union_area2 = (union_area1 - inter_area)
        return inter_area / np.maximum(union_area2, 1)

    def check_is_first(self, cls):
        if self.cfg.get('pair_method', None) is not None:
            if not 'first' in self.cfg['pair_method']:
                return cls > 0
            else:
                return cls in self.cfg['pair_method']['first']
        return cls > 0

    def check_is_second(self, cls):
        if self.cfg.get('pair_method', None) is not None:
            if not 'second' in self.cfg['pair_method']:
                return cls > 0
            else:
                return cls in self.cfg['pair_method']['second']
        return cls > 0
    
    def generate_pair(self, rois, input):
        '''
        return value pair_rois: [N,K] batch_ix, rois_1 idx, rois_2 idx
        '''
        rois = to_np_array(rois)
        B = max(rois[:, 0].astype(np.int32)) + 1
        pair_rois = []
        pair_position = []
        rois1 = []
        rois2 = []
        nrois = np.array(copy.deepcopy(rois))
        for b_ix in range(B):
            rois1_ix = []
            rois2_ix = []
            b_first_cnt = 0
            b_second_cnt = 0
            for i in range(nrois.shape[0]):
                # All body bbox
                if int(nrois[i,0]) == b_ix and self.check_is_first(int(rois[i, 6])):
                    rois1_ix.append(i)
                    b_first_cnt += 1
                    if self.cfg.get('pre_top_n', None):
                        if b_first_cnt > self.cfg['pre_top_n']:
                            break
                # All the other bbox
                if int(nrois[i,0]) == b_ix and self.check_is_second(int(rois[i, 6])):
                    rois2_ix.append(i)
                    b_second_cnt += 1
                    if self.cfg.get('pre_top_n', None):
                        if b_second_cnt > self.cfg['pre_top_n']:
                            break
            # For avoiding runtime error
            if len(rois1_ix) == 0:
                rois1_ix.append(0)
            if len(rois2_ix) == 0:
                rois2_ix.append(0)
            rois1_ix = np.array(rois1_ix)
            rois2_ix = np.array(rois2_ix)
            N = len(rois1_ix)
            M = len(rois2_ix)
            batch_pair_rois = np.zeros((N * M, 3))
            batch_pair_position = np.zeros((N * M, 5))
            batch_pair_rois[:, 1: 3] = np.stack(np.meshgrid(rois1_ix, rois2_ix), axis=2).reshape(-1,2)
            batch_pair_rois[:, 0] = b_ix
            batch_pair_position[:, 0] = b_ix
            batch_rois1 = nrois[batch_pair_rois[:,1].astype(np.int32)]
            batch_rois2 = nrois[batch_pair_rois[:,2].astype(np.int32)]
            batch_pair_position[:, 1] = np.minimum(batch_rois1[:, 1], batch_rois2[:, 1])
            batch_pair_position[:, 2] = np.minimum(batch_rois1[:, 2], batch_rois2[:, 2])
            batch_pair_position[:, 3] = np.maximum(batch_rois1[:, 3], batch_rois2[:, 3])
            batch_pair_position[:, 4] = np.maximum(batch_rois1[:, 4], batch_rois2[:, 4])
            pair_rois.append(batch_pair_rois)
            pair_position.append(batch_pair_position)
            rois1.append(batch_rois1)
            rois2.append(batch_rois2)
        pair_rois = np.vstack(pair_rois)
        pair_position = np.vstack(pair_position)
        rois1 = np.vstack(rois1)
        rois2 = np.vstack(rois2)
        pair_filter = []
        position_filter = []
        rois1_filter = []
        rois2_filter = []
        # pre-process, suppress the pairs whose object is far away from the body bbox
        ratio = self.cfg.get('make_pair_ratio', None)
        for i, pair in enumerate(pair_rois):
            idx1 = int(pair[1])
            idx2 = int(pair[2])
            if ratio is not None:
                body_x1, body_y1, body_x2, body_y2 = rois[idx1][1:5]
                w = body_x2 - body_x1
                h = body_y2 - body_y1
                body_x1 = max(body_x1 - w * ratio, 0)
                body_y1 = max(body_y1 - h * ratio, 0)
                body_x2 = body_x2 + w * ratio
                body_y2 = body_y2 + h * ratio
                xmax = max(body_x1, rois[idx2][1])
                xmin = min(body_x2, rois[idx2][3])
                ymax = max(body_y1, rois[idx2][2])
                ymin = min(body_y2, rois[idx2][4])
                   
            else:
                xmax = max(rois[idx1][1], rois[idx2][1])
                xmin = min(rois[idx1][3], rois[idx2][3])
                ymax = max(rois[idx1][2], rois[idx2][2])
                ymin = min(rois[idx1][4], rois[idx2][4])
            
            cross = max(xmin - xmax, 0) * max(ymin - ymax, 0)
            area = (rois[idx2][3] - rois[idx2][1]) * (rois[idx2][4] - rois[idx2][2])
            ioa = cross / (area + 0.1)
            label1 = rois1[i][6].astype(np.int32)
            label2 = rois2[i][6].astype(np.int32)
            # IOA filter 
            if (ioa > self.cfg.get('ioa_threshold', -1) or i == 0):
                if self.cfg.get('pre_filter', None) is None or (str(label1) in self.pre_filter and label2 in self.pre_filter[str(label1)]) or i == 0:
                    pair_filter.append(pair)
                    position_filter.append(pair_position[i])
                    rois1_filter.append(rois1[i])
                    rois2_filter.append(rois2[i])
            elif self.training:
                bbox1_x = np.array([rois[idx1][1], rois[idx1][3]])
                bbox1_y = np.array([rois[idx1][2], rois[idx1][4]])
                sub1 = np.min(
                        np.array([np.abs(bbox1_x - rois[idx2][1]), np.abs(bbox1_x - rois[idx2][3])])
                        )
                sub2 = np.min(
                        np.array([np.abs(bbox1_y - rois[idx2][4]), np.abs(bbox1_y - rois[idx2][4])])
                        )
                sub = min(sub1, sub2)
                if sub < 100 and self.cfg.get('ioa_random_ratio', 0.0) > random.random():
                    pair_filter.append(pair)
                    position_filter.append(pair_position[i])
                    rois1_filter.append(rois1[i])
                    rois2_filter.append(rois2[i])
        if self.cfg.get('ioa_threshold', None) is not None or self.cfg.get('pre_filter', None) is not None:
            pair_rois = np.array(pair_filter)
            pair_position = np.array(position_filter)
            rois1 = np.array(rois1_filter)
            rois2 = np.array(rois2_filter)
        return pair_rois, rois1, rois2, pair_position

    def print_param(self, feature, file_path):
        '''
        print param during tocaffe method
        '''
        return
        res_file = open(file_path, 'w')
        temp_feature = feature.view(-1)
        res_file.write(str(temp_feature.shape) + '\n')
        for i in range(len(temp_feature)):
            res_file.write(str(temp_feature[i].cpu().detach().numpy().tolist()) + '\n')
        res_file.close()
    
    def create_table(self, pair_rois):
        '''
        map two rois idx to pair idx
        '''
        roiTable = {}
        N = pair_rois.shape[0]
        for i in range(N):
            x = int(pair_rois[i][1])
            y = int(pair_rois[i][2])
            roiTable[(x,y)] = i
        return roiTable

    def forward(self, input):
        prefix = 'PairWiseNet'
        mode = input.get('runner_mode', 'val')
        self.cfg = copy.deepcopy(self.origin_cfg)
        if mode in self.cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))
        output = {}
        if self.training:
            if len(input['gt_assos'][0]) == 0:
                sv = 1
            elif input['gt_assos'][0][0][2] > 0:
                sv = 1
            elif self.cfg.get('ignore_false_example', False):
                sv = 0
            else:
                sv = 1
            if not 'dt_bboxes' in input or input['dt_bboxes'].shape[0] == 0:
                input['dt_bboxes'] = torch.HalfTensor([[0, 0, 0, 0, 0, 0.0, 1]]).cuda()
            loss, acc, predict_vector, predict_target = self.get_loss(input)
            if predict_vector is not None:
                output[prefix + '.predict_vector'] = predict_vector
                output[prefix + '.predict_target'] = predict_target
            for k, v in loss.items():
                output[prefix + k] = v * sv
            for k, v in acc.items():
                output[prefix + k] = v
        else:
            if not 'dt_bboxes' in input or input['dt_bboxes'].shape[0] == 0:
                if self.tocaffe:
                    input['dt_bboxes'] = torch.FloatTensor([[0, 0, 0, 0, 0, 0.0, 1]])
                else:
                    input['dt_bboxes'] = torch.HalfTensor([[0, 0, 0, 0, 0, 0.0, 1]]).cuda()
            assos, pred_cls = self.get_assos(input)
            if self.tocaffe:
                output[prefix + '.blobs.classification'] = pred_cls
            output['dt_bboxes'] = input['dt_bboxes']
            first_bbox = output['dt_bboxes'].clone()
            twice_pro = self.cfg.get('twice_match', False)
            output['dt_assos'] = assos[assos[:, -1] >= 0.5] if twice_pro else assos
            if self.cfg.get('post_processor', None) is not None:
                processor = SingleFramePostProcessorFactory.create(self.cfg['post_processor'])
                output = processor.process(output)
                if twice_pro:
                    input['dt_bboxes'] = torch.from_numpy(output['next_dt_bbox']).cuda()
                    self.cfg['ioa_threshold'] = -1
                    assos, pred_cls = self.get_assos_union(input)
                    first_assos = output['dt_assos'].copy()
                    output['dt_bboxes'] = first_bbox
                    output['dt_assos'] = assos[assos[:, -1] >= 0.2]
                    output = processor.process(output)
                    if output['dt_assos'].shape[0] == 0:
                        output['dt_assos'] = first_assos
                    else:
                        output['dt_assos'] = np.vstack([output['dt_assos'], first_assos])
        return output

    def roi_pooling(self, rois, x, stride, image_info):
        #pdb.set_trace()
        feature = self.roipool(rois[:,0:5], x, stride)
        if self.keep_origin_feature and self.pre_fc is None:
            return feature
        c = feature.numel() // feature.shape[0]
        feature = feature.view(-1, c).contiguous()
        if self.pre_fc is not None:
            feature = self.pre_fc(feature)
        return feature

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, image_info):
        #pdb.set_trace()
        mlvl_pred_feature = []
        for lvl_idx in levels:
            #logger.info(str(lvl_idx) + " " + str(x_rois[lvl_idx].shape[0]))
            if x_rois[lvl_idx].shape[0] > 0:
                rois = x_rois[lvl_idx]
                feature = x_features[lvl_idx]
                stride = x_strides[lvl_idx]
                pred_feature = self.roi_pooling(rois, feature, stride, image_info)
                mlvl_pred_feature.append(pred_feature)
        pred_feature = torch.cat(mlvl_pred_feature, dim=0)
        return pred_feature
        
    def extract_feature(self, rois, input, keep_rois=False):
        '''
        rois is original rois
        '''
        x_features = input['features']
        x_strides = input['strides']
        image_info = input['image_info']
        for i in range(len(x_features)):
            feature = x_features[i]
            if self.tocaffe:
                self.print_param(feature, 'fpn' +  str(i) + '.txt')
        if self.cfg.get('fpn', None):
            fpn = self.cfg['fpn']
            if self.tocaffe and not self.training:
                mlvl_rois, recover_inds = [rois] * len(fpn['fpn_levels']), None
            else:
                if not keep_rois:
                    mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois, original_inds=True)
                    rois = to_np_array(rois[recover_inds])
                else:
                    mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
                    rois = to_np_array(rois)
            pred_feature = self.mlvl_predict(
                mlvl_rois, x_features, x_strides, fpn['fpn_levels'], image_info)
            if keep_rois and not self.tocaffe:
                pred_feature = pred_feature[recover_inds]
        return rois, pred_feature
    
    def extract_relation_feature(self, rois, pair_rois, pair_position, input):
        rois1 = copy.deepcopy(rois)
        rois1, pred_rois1_feature = self.extract_feature(rois1, input, keep_rois=True)
        if self.tocaffe:
            # use CPU method
            rois1_feature = pred_rois1_feature.index_select(0, torch.LongTensor(pair_rois[:, 1])).contiguous()
            rois2_feature = pred_rois1_feature.index_select(0, torch.LongTensor(pair_rois[:, 2])).contiguous()
        else:
            rois1_feature = pred_rois1_feature.index_select(0, torch.cuda.LongTensor(pair_rois[:, 1])).contiguous()
            rois2_feature = pred_rois1_feature.index_select(0, torch.cuda.LongTensor(pair_rois[:, 2])).contiguous()
        if self.predict_kernel:
            rois2_feature = self.ibconv(rois1_feature, rois1, pair_rois, input)

        # rois1_feature is body feature
        # rois2_feature is object feature
        pred_human_feature = rois1_feature
        features = [rois1_feature, rois2_feature] 
        if self.union_box:
            _, union_feature = self.extract_feature(torch.Tensor(pair_position).type_as(rois1_feature), input, keep_rois=True)
            features.append(union_feature)
        if not self.cfg.get('position', None) is None:
            # add position feature
            position_feature = PositionTransform(rois1[pair_rois[:,1].astype(np.int32)], rois1[pair_rois[:,2].astype(np.int32)], input['image_info'])
            if self.cfg['position'] == 'embedding' or self.cfg['position'] == 'embedding app':
                position_feature = PositionEmbedding(position_feature, 256)
                if self.cfg['position'] == 'embedding app':
                    app_feature = self.position_relu(self.position_fc1(pred_human_feature))
                    app_feature = self.position_relu(self.position_fc2(app_feature))
                    app_feature = self.position_relu(self.position_fc3(app_feature))
                    position_feature = position_feature * app_feature

            features.append(position_feature)
        if self.keep_origin_feature:
            return features
        pred_pair_feature = torch.cat(features, dim=1)
        if self.tocaffe:
            self.print_param(pred_pair_feature, 'concat167.txt')
        return pred_pair_feature

    def binary_predict(self, x, rois1=None, rois2=None, union_rois=None):
        raise NotImplementedError

    def predict(self, x, rois1=None, rois2=None, union_rois=None, input=None):
        raise NotImplementedError

    def load_box(self, path):
        # load given bbox 
        fin = open(path, "r")
        contents = fin.readlines()
        rois = []
        for line in contents:
            anno = json.loads(line)
            tmp = [0]
            box = anno["bbox"]
            for item in box:
                #tmp.append(item * 13.0 / 40.0)
                tmp.append(item)
            tmp.append(anno["score"])
            label = anno["label"]
            # reassign bbox label 
            tmp.append(label)
            rois.append(tmp)
        rois = torch.cuda.FloatTensor(rois)
        return rois

    def get_assos_union(self, input):
        image_info = input['image_info']
        rois = input['dt_bboxes']
        #rois = input['dt_bboxes'][:self.cfg['pre_top_n']]
        #rois = self.load_box("/mnt/lustre/share/zhangmingyuan/to_ycj/0808/bbox.json")
        tmp_rois = copy.deepcopy(rois)
        pair_rois, rois1, rois2, pair_position = self.generate_pair(tmp_rois, input)
            
        self.union_box = True
        pred_pair_feature = self.extract_relation_feature(rois, pair_rois, pair_position, input)
        union_feature = torch.cat([pred_pair_feature[:, 3136*2:3136*3], 
            pred_pair_feature[:, 3136:3136*2], pred_pair_feature[:, 3136*3:]], dim=1)
        pred_cls = self.predict(union_feature, rois1, rois2, pair_position, input)
        self.union_box = False
        if self.cfg.get('cls_type', 'softmax') == 'softmax':
            pred_cls = F.softmax(pred_cls, dim=1)
        if self.tocaffe:
            self.print_param(pred_cls, 'softmax.txt')
        assos = predict_assos(rois, pair_rois, pred_cls, image_info, self.cfg, self.tocaffe)
        return assos, pred_cls

    def get_assos(self, input):
        image_info = input['image_info']
        rois = input['dt_bboxes']
        #rois = input['dt_bboxes'][:self.cfg['pre_top_n']]
        #rois = self.load_box("/mnt/lustre/share/zhangmingyuan/to_ycj/0808/bbox.json")
        tmp_rois = copy.deepcopy(rois)
        pair_rois, rois1, rois2, pair_position = self.generate_pair(tmp_rois, input)
        pred_pair_feature = self.extract_relation_feature(rois, pair_rois, pair_position, input)
        pred_cls = self.predict(pred_pair_feature, rois1, rois2, pair_position, input)

        if self.cfg.get('cls_type', 'softmax') == 'softmax':
            pred_cls = F.softmax(pred_cls, dim=1)
        if self.tocaffe:
            self.print_param(pred_cls, 'softmax.txt')
        assos = predict_assos(rois, pair_rois, pred_cls, image_info, self.cfg, self.tocaffe)
        return assos, pred_cls

    def assign(self, rois, gt_bboxes):
        # when use rpn output for asso prediction, we re-label the bbox cls by calculate the IoU between dt and gt
        # 1 is body bbox, 2 is object bbox 
        N = rois.shape[0]
        flag = False
        for i in range(N):
            b_ix = int(rois[i][0])
            gt = to_np_array(gt_bboxes[b_ix]).astype(np.int32)
            idx = np.where(gt[:, 4] == 1)
            gt = gt[idx]
            if gt.shape[0] == 0:
                if flag:
                    rois[i][6] = 2
                flag = True
                continue
            iou = cal_iou(to_np_array(rois[i][1:5]), gt)
            iou = np.max(iou)
            if iou < 0.5:
                if flag:
                    rois[i][6] = 2
                flag = True
        return rois

    def append_gt_bboxes(self, rois, gt_bboxes, image_info):
        B = len(image_info)
        new_rois = []
        for b_ix in range(B):
            gt_bbox = gt_bboxes[b_ix]
            N = gt_bbox.shape[0]
            nrois = torch.zeros((N, 7))
            nrois = nrois.type_as(rois)
            nrois[:, 0] = b_ix
            nrois[:, 1: 5] = gt_bbox[:, :4]
            nrois[:, 5] = 1.0
            nrois[:, 6] = gt_bbox[:, 4]
            w = nrois[:, 3] - nrois[:, 1]
            h = nrois[:, 4] - nrois[:, 2]
            rand = np.random.rand(N, 4)
            ratio = self.cfg.get('gt_jitter_ratio', 0.0)
            nrois[:, 1] = nrois[:, 1] + ((rand[:, 0] - 0.5) * 2 * w * ratio).type_as(nrois)
            nrois[:, 2] = nrois[:, 2] + ((rand[:, 1] - 0.5) * 2 * h * ratio).type_as(nrois)
            nrois[:, 3] = nrois[:, 3] + ((rand[:, 2] - 0.5) * 2 * w * ratio).type_as(nrois)
            nrois[:, 4] = nrois[:, 4] + ((rand[:, 3] - 0.5) * 2 * h * ratio).type_as(nrois)
            nrois[:, 1:5] = clip_bbox(nrois[:, 1:5], image_info[b_ix])
            if N > 20 and not self.cfg.get('only_gt_bboxes', False):
                keep_ix = np.random.choice(N, size=20, replace=True)
                nrois = nrois.index_select(0, torch.cuda.LongTensor(keep_ix)).contiguous()
            new_rois.append(nrois)
        if not self.cfg.get('only_gt_bboxes', False):
            new_rois.append(rois)
        rois = torch.cat(new_rois)
        return rois

    def get_loss(self, input):
        image_info = input['image_info']
        rois = input.get('dt_bboxes', None)
        gt_assos = input['gt_assos']
        gt_bboxes = input.get('gt_bboxes', None)
        if self.cfg.get('append_gt_bboxes', False):
            rois = self.append_gt_bboxes(rois, gt_bboxes, image_info)
        #rois = self.assign(rois, gt_bboxes)
        ignore_regions = input.get('gt_ignores', None)
        tmp_rois = copy.deepcopy(rois)
        pair_rois, rois1, rois2, pair_position = self.generate_pair(tmp_rois, input)
        #pdb.set_trace()
        roiTable = self.create_table(pair_rois)
        inds, rcnn_gt_cls, normalizer = \
            compute_proposal_targets_gt(
                tmp_rois, pair_rois, self.num_classes, self.cfg, gt_bboxes, gt_assos, roiTable, image_info, 'rcnn', ignore_regions)
        tmp_pair_rois = pair_rois[inds]
        tmp_pair_position = pair_position[inds]
        tmp_rois1 = rois1[inds]
        tmp_rois2 = rois2[inds]
        pred_pair_feature = self.extract_relation_feature(rois, tmp_pair_rois, tmp_pair_position, input)
        pred_rcnn_cls = self.predict(pred_pair_feature, tmp_rois1, tmp_rois2, tmp_pair_position, input)
        if isinstance(pred_rcnn_cls, list):
            loss, acc = {}, {}
            for i, pred_cls in enumerate(pred_rcnn_cls):
                cur_loss, cur_acc = self.cal_loss(input, tmp_rois1, tmp_rois2, pred_cls, rcnn_gt_cls, str(i) + '_', self.cfg.get("cls_loss_scale", 1.0), self.binary_mask)
                loss.update(cur_loss)
                acc.update(cur_acc)
        else:
            loss, acc = self.cal_loss(input, tmp_rois1, tmp_rois2, pred_rcnn_cls, rcnn_gt_cls, "", self.cfg.get("cls_loss_scale", 1.0), self.binary_mask)
        predict_vector = None
        predict_target = None

        return loss, acc, predict_vector, predict_target

    def cal_loss(self, input, rois1, rois2, pred_rcnn_cls, rcnn_gt_cls, prefix, cls_loss_scale, binary_mask=None):
        loss = {}
        def f2(x):
            return Variable(torch.from_numpy(x)).cuda()
        rcnn_gt_cls = f2(rcnn_gt_cls).long()
        rcnn_cls = pred_rcnn_cls.float()
        if self.cfg.get('cls_type', 'softmax') == 'sigmoid':
            rcnn_gt_cls = rcnn_gt_cls.float()
        sigma = self.cfg.get('smooth_l1_sigma', 1.0)
        if self.cfg.get('ohem', None):
            rcnn_cls = rcnn_cls.view(-1).contiguous()
            rcnn_gt_cls = rcnn_gt_cls.view(-1).contiguous()
            cls_loss, _, idx = L.ohem_loss(
                self.cfg['ohem']['batch_size'],
                rcnn_cls,
                rcnn_gt_cls,
                None,
                None,
                self.cfg.get('cls_type', 'softmax'),
                smooth_l1_sigma=sigma)
            rcnn_cls = rcnn_cls[idx]
            rcnn_gt_cls = rcnn_gt_cls[idx]
        elif self.cfg.get('focal_loss', None) is not None:
            cls_loss, acc = L.get_focal_loss(rcnn_cls, rcnn_gt_cls, rcnn_gt_cls.shape[0], self.num_classes, self.cfg['focal_loss'])
            #pdb.set_trace()
        elif self.cfg.get('cls_type', 'softmax') == 'softmax':
#（N， 6） rois2 
            # rcnn_cls
            weight = self.cfg.get('cls_weight', None)
            weight = torch.tensor(weight).float().cuda() if weight is not None else None
            cls_loss = F.cross_entropy(rcnn_cls, rcnn_gt_cls, weight=weight)
        else:
            if self.binary_mask:
                N, C = rcnn_cls.shape
                for i in range(N):
                    scale = self.binary_scale[rois2[i][6]]
                    scale = scale.type_as(rcnn_cls)
                    rcnn_cls[i] = rcnn_cls[i] * scale
            cls_loss = F.binary_cross_entropy_with_logits(rcnn_cls, rcnn_gt_cls)
        loss.update({'.' + prefix + 'cls_loss': cls_loss * cls_loss_scale})
        if self.cfg.get('cls_type', 'softmax') == 'softmax':
            acc = {'.' + prefix + 'accuracy': A.accuracy(rcnn_cls, rcnn_gt_cls)[0]}
        else:
            acc = {'.' + prefix + 'accuracy': A.binary_accuracy(rcnn_cls, rcnn_gt_cls)[0]}
        return loss, acc


class GeneralAssociation(PairWiseNet):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 cfg,
                 initializer=None):
        super(GeneralAssociation, self).__init__(inplanes, num_classes, cfg)
        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(self.mdim, feat_planes)
        self.fc7 = nn.Linear(feat_planes, feat_planes)
        self.fc_rcnn_cls = nn.Linear(feat_planes + self.cls_mdim, num_classes)
                

    def predict(self, x, rois1=None, rois2=None, union_rois=None, input=None):
        '''
            x: feature to pool
        '''
        feature = self.relu(self.fc6(x))
        feature = self.relu(self.fc7(feature))
        pred_cls = self.fc_rcnn_cls(feature)
        if self.tocaffe:
            self.print_param(x, 'concat_feature.txt')
            self.print_param(pred_cls, 'pred_feature.txt')
        return pred_cls

class GeneralAssociation_union(PairWiseNet):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 cfg,
                 initializer=None):
        super(GeneralAssociation_union, self).__init__(inplanes, num_classes, cfg)
        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)
        self.fc6_2 = nn.Linear(self.mdim, feat_planes)
        self.fc7_2 = nn.Linear(feat_planes, feat_planes)
        self.fc_rcnn_cls_2 = nn.Linear(feat_planes + self.cls_mdim, num_classes)
                

    def predict(self, x, rois1=None, rois2=None, union_rois=None, input=None):
        '''
            x: feature to pool
        '''
        feature = self.relu(self.fc6_2(x))
        feature = self.relu(self.fc7_2(feature))
        pred_cls = self.fc_rcnn_cls_2(feature)
        if self.tocaffe:
            self.print_param(x, 'concat_feature.txt')
            self.print_param(pred_cls, 'pred_feature.txt')
        return pred_cls
        
