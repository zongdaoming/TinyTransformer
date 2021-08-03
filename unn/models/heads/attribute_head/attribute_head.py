import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .... import extensions as E
from ...initializer import init_weights_normal
from ...initializer import initialize_from_cfg
from ....utils.bn_helper import setup_bn, rollback_bn, FREEZE
from ..utils import accuracy as A
from ..utils import loss as L
from ..utils.assigner import map_rois_to_level
from ....utils.bn_helper import setup_bn, rollback_bn, FREEZE
from .attribute import proposal_targets

__all__ = ['FC']

logger = logging.getLogger('global')


class AttributeNet(nn.Module):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
    """

    def __init__(self, inplanes, num_classes, cfg):
        super(AttributeNet, self).__init__()

        self.origin_cfg = copy.deepcopy(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.tocaffe = self.cfg.get('tocaffe', False)
        self.cfg['num_classes'] = num_classes

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        self.roipool = E.build_generic_roipool(cfg['roipooling'])
        self.pool_size = cfg['roipooling']['pool_size']
        self.output_predict = self.cfg.get('output_predict', False)

    def predict(self, rois, x, stride):
        """
        Arguments:
            rois (FloatTensor): rois in a sinle layer
            x (FloatTensor): features in a single layer
            stride: stride for current layer
        """
        raise NotImplementedError

    def forward(self, input):
        prefix = 'AttributeNet'
        self.cfg = copy.deepcopy(self.origin_cfg)
        mode = input.get('runner_mode', 'val')
        if mode in self.cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))

        output = {}
        if self.training:
            sample_record, cls_loss, acc, predict_vector, predict_target = self.get_loss(input)
            loss_scale = self.cfg.get('loss_scale', 1.0)
            output[prefix + '.cls_loss'] = cls_loss * loss_scale
            output[prefix + '.accuracy'] = acc
            if predict_vector is not None:
                output[prefix + '.predict_vector'] = F.sigmoid(predict_vector)
                output[prefix + '.predict_target'] = predict_target
                
        else:
            if len(input['dt_bboxes']) == 0:
                return output
            attris = self.get_attris(input)
            output['dt_attris'] = attris
        return output

    def mlvl_predict(self, x_rois, x_features, x_strides, levels):
        """Predict results level by level"""
        mlvl_cls_pred = []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                rois = x_rois[lvl_idx]
                feature = x_features[lvl_idx]
                stride = x_strides[lvl_idx]
                cls_pred = self.predict(rois, feature, stride)
                mlvl_cls_pred.append(cls_pred)
        cls_pred = torch.cat(mlvl_cls_pred, dim=0)
        return cls_pred

    def get_head_output(self, rois, features, strides, cfg):
        if cfg.get('fpn', None):
            # assign rois and targets to each level
            fpn = self.cfg['fpn']
            if self.tocaffe and not self.training:
                # make sure that all pathways included in the computation graph
                mlvl_rois, recover_inds = [rois] * len(fpn['fpn_levels']), None
            else:
                mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'])
            rois = torch.cat(mlvl_rois, dim=0)
            # cls_pred = cls_pred[recover_inds]
            # loc_pred = loc_pred[recover_inds]
        else:
            assert len(features) == 1 and len(strides) == 1, \
                'please setup `fpn` first if you want to use pyramid features'
            cls_pred = self.predict(rois, features[0], strides[0])
            recover_inds = torch.arange(rois.shape[0], device=rois.device)
        return rois, cls_pred.float(), recover_inds

    def get_loss(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']
        gt_bboxes = input['gt_bboxes']
        gt_attris = input['gt_attris']
        ignore_regions = input.get('gt_ignores', None)

        sample_record, sampled_rois, cls_target = proposal_targets(
            rois, gt_bboxes, gt_attris, image_info, self.cfg, ignore_regions)
        rois, cls_pred, recover_inds = self.get_head_output(sampled_rois, features, strides, self.cfg)
        cls_pred = cls_pred[recover_inds]
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred.float(), cls_target.float())
        acc = A.binary_accuracy(cls_pred, cls_target)[0]
        predict_vector = None
        predict_target = None
        if self.output_predict:
            predict_vector = F.sigmoid(cls_pred)
            predict_target = cls_target
        return sample_record, cls_loss, acc, predict_vector, predict_target

    def get_attris(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']

        rois, cls_pred, recover_inds = self.get_head_output(rois, features, strides, self.cfg)
        cls_pred = F.sigmoid(cls_pred)
        cls_pred = cls_pred[recover_inds]
        return cls_pred


class FC(AttributeNet):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        feat_planes (int): channels of intermediate features
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
        initializer (dict): config for module parameters initialization
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        super(FC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
        self.fc7 = nn.Linear(feat_planes, feat_planes)

        self.fc_rcnn_cls = nn.Linear(feat_planes, num_classes)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls, 0.01)

    def predict(self, rois, x, stride):
        x = self.roipool(rois, x, stride)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        cls_pred = self.fc_rcnn_cls(x)
        return cls_pred


