import copy
import logging

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .... import extensions as E

from ..utils.assigner import map_rois_to_level

logger = logging.getLogger('global')
def build_ibconv(inplanes, pool_size, cfg):
    method = cfg['predict_kernel']['method']
    if method == 'conv':
        return IBConv(inplanes, pool_size, cfg)
    elif method == 'deform':
        return IBDeformConv(inplanes, pool_size, cfg)
    else:
        logger.error('Unrecognized method')

class IBBaseConv(nn.Module):

    def __init__(self, inplanes, pool_size, cfg):
        super(IBBaseConv, self).__init__()
        self.feat_dim = pool_size * pool_size * inplanes
        self.group_num = cfg['predict_kernel']['group']
        self.ratio = cfg['predict_kernel']['ratio']
        self.kernel_size = cfg['predict_kernel']['kernel_size']
        self.freq = cfg['predict_kernel']['freq']
        self.inplanes = inplanes
        self.pool_size = pool_size
        self.cfg = copy.deepcopy(cfg)
        self.roipool = E.build_generic_roipool(cfg['roipooling'])

    def roi_pooling(self, rois, x, stride):
        feature = self.roipool(rois[:, 0:5], x, stride)
        c = feature.numel() // feature.shape[0]
        feature = feature.view(-1, c)
        return feature.contiguous()

    def mlvl_predict(self, x_rois, x_features, x_strides, levels):
        mlvl_pred_feature = []
        for lvl_idx in levels:
            if x_rois[lvl_idx].shape[0] > 0:
                rois = x_rois[lvl_idx]
                feature = x_features[lvl_idx]
                stride = x_strides[lvl_idx]
                pred_feature = self.roi_pooling(rois, feature, stride)
                mlvl_pred_feature.append(pred_feature)
        pred_feature = torch.cat(mlvl_pred_feature, dim=0)
        return pred_feature

    def transform(self, instance, feature):
        pass

    def forward(self, rois1_feature, rois, pair_rois, input):
        mode = input.get('runner_mode', 'val')
        if mode == 'train':
            freq = self.freq
        else:
            freq = 1
        rois1_idx = np.unique(pair_rois[:, 1]).astype(np.int32)
        fpn = self.cfg['fpn']
        strides = input['strides']
        rois2_feature = torch.zeros_like(rois1_feature)
        cnt = 0
        for idx1 in rois1_idx:
            idx = np.where(pair_rois[:, 1] == idx1)[0]
            cnt += len(idx)
            instance = rois1_feature[idx1]
            new_features = []
            self.setup_conv = False
            value = random.random()
            if value <= freq:
                for feature in input['features']:
                    new_features.append(self.transform(instance, feature))
            else:
                new_features = input['features']
            rois2_idx = pair_rois[idx, 2].astype(np.int32)
            tmp_rois = rois[rois2_idx]
            mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], torch.from_numpy(tmp_rois).cuda())
            pred_feature = self.mlvl_predict(mlvl_rois, new_features, strides, fpn['fpn_levels'])
            pred_feature = pred_feature[recover_inds]
            rois2_feature[idx] = pred_feature
        return rois2_feature


class IBConv(IBBaseConv):

    def __init__(self, inplanes, pool_size, cfg):
        super(IBConv, self).__init__(inplanes, pool_size, cfg)
        self.fc1 = nn.Linear(self.inplanes * self.pool_size * self.pool_size, self.inplanes * self.pool_size * self.pool_size // self.ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.inplanes * self.pool_size * self.pool_size // self.ratio, self.inplanes * self.inplanes // self.group_num * self.kernel_size * self.kernel_size)

    def get_kernel(self, instance):
        feature = self.fc1(instance)
        feature = self.relu(feature)
        feature = self.fc2(feature)
        feature = feature.view(self.inplanes, self.inplanes // self.group_num, self.kernel_size, self.kernel_size).contiguous()
        return feature

    def transform(self, instance, feature):
        if not self.setup_conv:
            self.conv_kernel = self.get_kernel(instance)
            self.setup_conv = True
        new_feature = F.conv2d(feature, self.conv_kernel, padding=self.kernel_size // 2, groups=self.group_num)
        return new_feature


class IBDeformConv(IBBaseConv):

    def __init__(self, inplanes, pool_size, cfg):
        super(IBDeformConv, self).__init__(inplanes, pool_size, cfg)
        self.bottleneck = self.cfg['predict_kernel']['bottleneck']
        self.fc1 = nn.Linear(self.inplanes * self.pool_size * self.pool_size, self.inplanes * self.pool_size * self.pool_size // self.ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.inplanes * self.pool_size * self.pool_size // self.ratio, 2 * self.inplanes * self.inplanes // self.group_num * self.kernel_size * self.kernel_size // self.bottleneck // self.bottleneck)
        self.conv1 = nn.Conv2d(inplanes, inplanes // self.bottleneck, kernel_size=1)
        self.conv2 = E.DeformableConv(inplanes // self.bottleneck, inplanes, kernel_size=3, padding=1)

    def get_kernel(self, instance):
        feature = self.fc1(instance)
        feature = self.relu(feature)
        feature = self.fc2(feature)
        feature = feature.view(2 * self.inplanes // self.bottleneck, self.inplanes // self.group_num // self.bottleneck, self.kernel_size, self.kernel_size).contiguous()
        
        return feature

    def transform(self, instance, feature):
        if not self.setup_conv:
            self.conv_kernel = self.get_kernel(instance)
            self.setup_conv = True
        new_feature = self.conv1(feature)
        offset = F.conv2d(new_feature, self.conv_kernel, padding=self.kernel_size // 2, groups=self.group_num)
        new_feature = self.conv2(new_feature, offset)
        return new_feature
