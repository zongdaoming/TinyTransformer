import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .... import extensions as E
from ...initializer import init_weights_normal
from ...initializer import initialize_from_cfg
from ..utils import accuracy as A
from ..utils import loss as L
from ..utils.assigner import map_rois_to_level
from .bbox import predict_bboxes
from .bbox import refine_bboxes
from .bbox import proposal_targets_with_gt_flag

__all__ = ['FC']

logger = logging.getLogger('global')


class CascadeBboxNet(nn.Module):
    """
    Cascade boxes prediction and refinement.
    """
    def __init__(self, inplanes, num_classes, cfg):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
        """
        super(CascadeBboxNet, self).__init__()

        self.origin_cfg = copy.deepcopy(cfg)  # keep the original configuration
        self.cfg = copy.deepcopy(cfg)  # runtime configuration
        self.tocaffe = self.cfg.get('tocaffe', False)
        self.cfg['num_classes'] = num_classes
        self.num_stage = self.cfg.get('num_stage', 0)
        self.stage_weights = self.cfg.get('stage_weights', None)
        self.test_stage = self.cfg.get('test_stage', 0)
        self.test_ensemble = self.cfg.get('test_ensemble', True)

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        roipool = E.build_generic_roipool(cfg['roipooling'])
        self.roipool_list = nn.ModuleList()
        for i in range(self.num_stage):
            self.roipool_list.append(roipool)
        self.pool_size = cfg['roipooling']['pool_size']

    def predict(self, rois, x, stride, stage):
        """
        Arguments:
            rois (FloatTensor): rois in a sinle layer
            x (FloatTensor): features in a single layer
            stride: stride for current layer
        """
        raise NotImplementedError

    def forward(self, input):
        """
        Arguments:
            - input (:obj:`dict`): data from prev module

        Returns:
            - output (:obj:`dict`): output k, v is different for training and testing

        Input example::

            # input
            {
                # (list of FloatTensor): input feature layers,
                # for C4 from backbone, others from FPN
                'features': ..,
                # (list of int): input feature layers,
                # for C4 from backbone, others from FPN
                'strides': [],
                # (list of FloatTensor)
                # [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
                'image_info': [],
                # (FloatTensor): boxes from last module,
                # [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
                'dt_bboxes': [],
                # (list of FloatTensor or None): [B] [num_gts, 5] (x1, y1, x2, y2, label)
                'gt_bboxes': [],
                # (list of FloatTensor): [B] [num_igs, 4] (x1, y1, x2, y2)
                'gt_ignores': []
            }

        Output example::

            # training output
            # 0 <= i < num_stage
            {'BboxNet.cls_loss{i}': <tensor>, 'BboxNet.loc_loss_{i}': <tensor>, 'BboxNet.accuracy_{i}': <tensor>}

            # testing output
            {
                # (FloatTensor), predicted boxes [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
                'dt_bboxes': <tensor>,
            }
        """
        prefix = 'BboxNet'
        self.cfg = copy.deepcopy(self.origin_cfg)
        mode = input.get('runner_mode', 'val')
        if mode in self.cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))
        #self.cfg.update(self.cfg.get("train" if self.training else "test", {}))

        output = {}
        if self.training:
            stage_sample_record, stage_cls_loss, stage_loc_loss, stage_acc = self.get_loss(input)
            for i in range(self.num_stage):
                # output['sample_record_' + str(i)] = stage_sample_record[i]
                output['sample_record'] = stage_sample_record[i]
                output[prefix + '.cls_loss_' + str(i)] = stage_cls_loss[i]
                output[prefix + '.loc_loss_' + str(i)] = stage_loc_loss[i]
                output[prefix + '.accuracy_' + str(i)] = stage_acc[i]
            if self.cfg.get('generate_bbox', True):
                bboxes, cls_pred, loc_pred = self.get_bboxes(input)
                output['dt_bboxes'] = bboxes
        else:
            bboxes, cls_pred, loc_pred = self.get_bboxes(input)
            output['dt_bboxes'] = bboxes
        return output

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, stage):
        """Predict results level by level"""
        mlvl_cls_pred, mlvl_loc_pred = [], []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                rois = x_rois[lvl_idx]
                feature = x_features[lvl_idx]
                stride = x_strides[lvl_idx]
                cls_pred, loc_pred = self.predict(rois, feature, stride, stage)
                mlvl_cls_pred.append(cls_pred)
                mlvl_loc_pred.append(loc_pred)
        cls_pred = torch.cat(mlvl_cls_pred, dim=0)
        loc_pred = torch.cat(mlvl_loc_pred, dim=0)
        return cls_pred, loc_pred

    def get_head_output(self, rois, features, strides, cfg, stage):
        """
        Assign rois to each level and predict

        Note:
            1.The recovering process is not supposed to be handled in this function,
              because ONNX DON'T support indexing;
            2.numerical type of cls_pred and loc_pred must be float for fp32 support !!!

        Returns:
            rois (FloatTensor): assigned rois
            cls_pred (FloatTensor, fp32): prediction of classification of assigned rois
            loc_pred (FloatTensor, fp32): prediction of localization of assigned rois
            recover_inds (LongTensor): indices of recovering input rois from assigned rois
        """
        if cfg.get('fpn', None):
            # assign rois and targets to each level
            fpn = self.cfg['fpn']
            mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred, loc_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'], stage)
            rois = torch.cat(mlvl_rois, dim=0)
            # cls_pred = cls_pred[recover_inds]
            # loc_pred = loc_pred[recover_inds]
        else:
            assert len(features) == 1 and len(strides) == 1, \
                'please setup `fpn` first if you want to use pyramid features'
            cls_pred, loc_pred = self.predict(rois, features[0], strides[0], stage)
            recover_inds = torch.arange(rois.shape[0], device=rois.device)
        return rois, cls_pred.float(), loc_pred.float(), recover_inds

    def get_cascade_stage_cfg(self, stage_i):
        cfg_i = copy.deepcopy(self.cfg)
        cfg_i['bbox_normalize'] = cfg_i['stage_bbox_normalize']
        cfg_i['bbox_normalize']['means'] = \
            cfg_i['stage_bbox_normalize']['means'][stage_i]
        cfg_i['bbox_normalize']['stds'] = \
            cfg_i['stage_bbox_normalize']['stds'][stage_i]

        if self.training:
            cfg_i['matcher'] = \
                cfg_i['train']['stage_matcher']
            cfg_i['matcher']['positive_iou_thresh'] = \
                cfg_i['train']['stage_matcher']['positive_iou_thresh'][stage_i]
            cfg_i['matcher']['negative_iou_thresh'] = \
                cfg_i['train']['stage_matcher']['negative_iou_thresh'][stage_i]
        return cfg_i

    def get_loss(self, input):
        """
        Arguments:
            input['features'] (list): input feature layers, for C4 from backbone, others from FPN
            input['strides'] (list): strides of input feature layers
            input['image_info'] (list of FloatTensor): [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
            input['dt_bboxes'] (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
            input['gt_bboxes'] (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
            input['gt_ignores'] (list of FloatTensor): [B, num_igs, 4] (x1, y1, x2, y2)

        Returns:
            sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)], saved for mask/keypoint head
            cls_loss, loc_loss, acc (FloatTensor)
        """
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)

        stage_sample_record = []
        stage_cls_loss = []
        stage_loc_loss = []
        stage_acc = []
        for cascade_i in range(self.num_stage):
            stage_weight = self.stage_weights[cascade_i]
            cascade_i_cfg = self.get_cascade_stage_cfg(cascade_i)
            # cls_target (LongTensor): [R]
            # loc_target (FloatTensor): [R, 4]
            # loc_weight (FloatTensor): [R, 4]
            sample_record, sampled_rois, cls_target, loc_target, loc_weight, gt_flags = proposal_targets_with_gt_flag(
                rois, gt_bboxes, image_info, cascade_i_cfg, ignore_regions)
            rois, cls_pred, loc_pred, recover_inds = self.get_head_output(sampled_rois, features, strides,
                                                                          cascade_i_cfg, cascade_i)
            cls_pred = cls_pred[recover_inds]
            loc_pred = loc_pred[recover_inds]

            cls_inds = cls_target
            if cascade_i_cfg.get('share_location', 'False'):
                cls_inds = cls_target.clamp(max=0)

            N = loc_pred.shape[0]
            loc_pred = loc_pred.reshape(N, -1, 4)
            inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
            loc_pred = loc_pred[inds, cls_inds].reshape(-1, 4)

            sigma = cascade_i_cfg['smooth_l1_sigma']
            if cascade_i_cfg.get('ohem', None):
                cls_loss, loc_loss = L.ohem_loss(
                    cascade_i_cfg['ohem']['batch_size'],
                    cls_pred,
                    cls_target,
                    loc_pred * loc_weight,
                    loc_target,
                    smooth_l1_sigma=sigma)
            else:
                cls_loss = F.cross_entropy(cls_pred, cls_target, ignore_index=-1)
                loc_loss = L.smooth_l1_loss(
                    loc_pred * loc_weight, loc_target, sigma=sigma, normalizer=cls_target.shape[0])
            acc = A.accuracy(cls_pred, cls_target)[0]

            # collect cascade stage loss and accuracy
            cls_loss = cls_loss * stage_weight
            loc_loss = loc_loss * stage_weight
            stage_sample_record.append(sample_record)
            stage_cls_loss.append(cls_loss)
            stage_loc_loss.append(loc_loss)
            stage_acc.append(acc)
            # refine bboxes before the last stage
            rois = rois[recover_inds]
            with torch.no_grad():
                rois = refine_bboxes(rois, cls_target, loc_pred, gt_flags, image_info, cascade_i_cfg)
        return stage_sample_record, stage_cls_loss, stage_loc_loss, stage_acc

    def get_bboxes(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']

        stage_scores = []
        for cascade_i in range(self.test_stage):
            cascade_i_cfg = self.get_cascade_stage_cfg(cascade_i)
            rois, cls_pred, loc_pred, recover_inds = self.get_head_output(rois, features, strides, cascade_i_cfg,
                                                                          cascade_i)
            rois = rois.detach()[recover_inds]
            cls_pred = cls_pred.detach()[recover_inds]
            loc_pred = loc_pred.detach()[recover_inds]
            cls_pred = F.softmax(cls_pred, dim=1)
            stage_scores.append(cls_pred)

            if cascade_i < self.test_stage - 1:
                gt_flags = rois.new_zeros(rois.shape[0])
                rois = refine_bboxes(
                    rois, cls_pred.argmax(dim=1), loc_pred.detach(), gt_flags, image_info, cascade_i_cfg)

        if self.test_ensemble:
            cls_pred = sum(stage_scores) / self.test_stage

        bboxes, score_for_grid = predict_bboxes(rois, cls_pred.detach(), loc_pred.detach(), image_info, cascade_i_cfg)
        return bboxes, cls_pred, loc_pred


class FC(CascadeBboxNet):
    """
    Use FC as head
    """
    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization

        `Cascade FC example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/cascade-rcnn-R50-FPN-1x.yaml#L136-172>`_
        """
        super(FC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)
        # self.fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
        # self.fc7 = nn.Linear(feat_planes, feat_planes)

        # self.fc_rcnn_cls = nn.Linear(feat_planes, num_classes)
        # if self.cfg.get('share_location', False):
        #     self.fc_rcnn_loc = nn.Linear(feat_planes, 4)
        # else:
        #     self.fc_rcnn_loc = nn.Linear(feat_planes, num_classes * 4)
        self.fc6_list = nn.ModuleList()
        self.fc7_list = nn.ModuleList()
        self.fc_rcnn_cls_list = nn.ModuleList()
        self.fc_rcnn_loc_list = nn.ModuleList()
        for i in range(self.num_stage):
            fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
            self.fc6_list.append(fc6)
            fc7 = nn.Linear(feat_planes, feat_planes)
            self.fc7_list.append(fc7)

            fc_rcnn_cls = nn.Linear(feat_planes, num_classes)
            self.fc_rcnn_cls_list.append(fc_rcnn_cls)
            if self.cfg.get('share_location', False):
                fc_rcnn_loc = nn.Linear(feat_planes, 4)
            else:
                fc_rcnn_loc = nn.Linear(feat_planes, num_classes * 4)
            self.fc_rcnn_loc_list.append(fc_rcnn_loc)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls_list, 0.01)
        init_weights_normal(self.fc_rcnn_loc_list, 0.001)

    def predict(self, rois, x, stride, stage):
        roipool = self.roipool_list[stage]
        fc6 = self.fc6_list[stage]
        fc7 = self.fc7_list[stage]
        fc_rcnn_cls = self.fc_rcnn_cls_list[stage]
        fc_rcnn_loc = self.fc_rcnn_loc_list[stage]
        x = roipool(rois, x, stride)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        x = self.relu(fc6(x))
        x = self.relu(fc7(x))
        cls_pred = fc_rcnn_cls(x)
        loc_pred = fc_rcnn_loc(x)
        return cls_pred, loc_pred
