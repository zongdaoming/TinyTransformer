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
from .bbox import predict_bboxes
from .bbox import proposal_targets
import pdb

__all__ = ['FC', 'Res5', 'RFCN']

logger = logging.getLogger('global')


class BboxNet(nn.Module):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
    """

    def __init__(self, inplanes, num_classes, cfg):
        super(BboxNet, self).__init__()

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
        
    def predict(self, rois, x, stride):
        """
        Arguments:
            rois (FloatTensor): rois in a sinle layer
            x (FloatTensor): features in a single layer
            stride: stride for current layer
        """
        raise NotImplementedError

    def forward(self, input):
        prefix = 'BboxNet'
        self.cfg = copy.deepcopy(self.origin_cfg)
        mode = input.get('runner_mode', 'val')
        if mode in self.cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))

        output = {}
        if self.training:
            sample_record, cls_loss, loc_loss, acc = self.get_loss(input)
            output['sample_record'] = sample_record
            output[prefix + '.cls_loss'] = cls_loss * self.cfg.get('cls_loss_scale', 1.0)
            if self.cfg.get('grid', None):
                output[prefix + '.loc_loss'] = loc_loss * 0
            else:
                output[prefix + '.loc_loss'] = loc_loss * self.cfg.get('loc_loss_scale', 1.0)
            output[prefix + '.accuracy'] = acc
            if self.cfg.get('generate_bbox', False):
                bboxes, cls_pred, loc_pred, selected_pred_cls = self.get_bboxes(input)
                output['dt_bboxes'] = bboxes
        else:
            bboxes, cls_pred, loc_pred, selected_pred_cls = self.get_bboxes(input)
            output['dt_bboxes'] = bboxes
            self.use_gt = False
            if self.use_gt:
                bboxes = input['gt_bboxes']
                rois = []
                for bz_id, bbox in enumerate(bboxes):
                    tmp = torch.zeros(bbox.shape[0], 7)
                    tmp[:, 0] = bz_id
                    tmp[:, 1: 5] = bbox[:, :4]
                    tmp[:, 5] = 1.0
                    tmp[:, 6] = bbox[:, -1]
                    rois.append(tmp)
                rois = torch.cat(rois, 0)
                #pdb.set_trace()
                rois = rois.cuda().half()
                output['dt_bboxes'] = rois
            # for grid rcnn
            output['pred_cls_prob'] = selected_pred_cls
            if self.tocaffe:
                output[prefix + '.blobs.classification'] = cls_pred
                output[prefix + '.blobs.localization'] = loc_pred
        return output

    def mlvl_predict(self, x_rois, x_features, x_strides, levels):
        """Predict results level by level"""
        mlvl_cls_pred, mlvl_loc_pred = [], []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                rois = x_rois[lvl_idx]
                feature = x_features[lvl_idx]
                stride = x_strides[lvl_idx]
                cls_pred, loc_pred = self.predict(rois, feature, stride)
                mlvl_cls_pred.append(cls_pred)
                mlvl_loc_pred.append(loc_pred)
        cls_pred = torch.cat(mlvl_cls_pred, dim=0)
        loc_pred = torch.cat(mlvl_loc_pred, dim=0)
        return cls_pred, loc_pred

    def get_head_output(self, rois, features, strides, cfg):
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
            if self.tocaffe and not self.training:
                # make sure that all pathways included in the computation graph
                mlvl_rois, recover_inds = [rois] * len(fpn['fpn_levels']), None
            else:
                mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred, loc_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'])
            rois = torch.cat(mlvl_rois, dim=0)
            # cls_pred = cls_pred[recover_inds]
            # loc_pred = loc_pred[recover_inds]
        else:
            assert len(features) == 1 and len(strides) == 1, \
                'please setup `fpn` first if you want to use pyramid features'
            cls_pred, loc_pred = self.predict(rois, features[0], strides[0])
            recover_inds = torch.arange(rois.shape[0], device=rois.device)
        return rois, cls_pred.float(), loc_pred.float(), recover_inds

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

        # cls_target (LongTensor): [R]
        # loc_target (FloatTensor): [R, 4]
        # loc_weight (FloatTensor): [R, 4]
        sample_record, sampled_rois, cls_target, loc_target, loc_weight = proposal_targets(
            rois, gt_bboxes, image_info, self.cfg, ignore_regions)
        rois, cls_pred, loc_pred, recover_inds = self.get_head_output(sampled_rois, features, strides, self.cfg)
        cls_pred = cls_pred[recover_inds]
        loc_pred = loc_pred[recover_inds]

        cls_inds = cls_target
        if self.cfg.get('share_location', 'False'):
            cls_inds = cls_target.clamp(max=0)

        N = loc_pred.shape[0]
        loc_pred = loc_pred.reshape(N, -1, 4)
        inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
        loc_pred = loc_pred[inds, cls_inds].reshape(-1, 4)

        sigma = self.cfg.get('smooth_l1_sigma', 3.0)
        if self.cfg.get('ohem', None):
            cls_loss, loc_loss, _ = L.ohem_loss(
                self.cfg['ohem']['batch_size'],
                cls_pred,
                cls_target,
                loc_pred * loc_weight,
                loc_target,
                smooth_l1_sigma=sigma)
        elif self.cfg.get('cls_loss', None) is 'balanced_l1_loss':
            weight = self.cfg.get('cls_weight', None)
            weight = torch.tensor(weight).float().cuda() if weight is not None else None
            cls_loss = F.cross_entropy(cls_pred, cls_target, ignore_index=-1, weight=weight)
            loc_loss = L.balanced_l1_loss(loc_pred * loc_weight, loc_target, normalizer=cls_target.shape[0])
        else:
            weight = self.cfg.get('cls_weight', None)
            weight = torch.tensor(weight).float().cuda() if weight is not None else None
            cls_loss = F.cross_entropy(cls_pred, cls_target, ignore_index=-1, weight=weight)
            loc_loss = L.smooth_l1_loss(loc_pred * loc_weight, loc_target, sigma=sigma, normalizer=cls_target.shape[0])
        acc = A.accuracy(cls_pred, cls_target)[0]
        return sample_record, cls_loss, loc_loss, acc

    def get_bboxes(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']

        rois, cls_pred, loc_pred, recover_inds = self.get_head_output(rois, features, strides, self.cfg)
        cls_pred = F.softmax(cls_pred, dim=1)
        bboxes, selected_pred_cls = predict_bboxes(rois, cls_pred.detach(), loc_pred.detach(), image_info, self.cfg)
        return bboxes, cls_pred, loc_pred, selected_pred_cls


class FC(BboxNet):
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
        if self.cfg.get('share_location', False):
            self.fc_rcnn_loc = nn.Linear(feat_planes, 4)
        else:
            self.fc_rcnn_loc = nn.Linear(feat_planes, num_classes * 4)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls, 0.01)
        init_weights_normal(self.fc_rcnn_loc, 0.001)

        self.freeze = self.cfg.get('freeze', False)
        if self.freeze:
            for name, module in self.named_children():
                for param in module.parameters():
                    param.requires_grad = False

    def predict(self, rois, x, stride):
        x = self.roipool(rois, x, stride)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        cls_pred = self.fc_rcnn_cls(x)
        loc_pred = self.fc_rcnn_loc(x)
        return cls_pred, loc_pred


class Res5(BboxNet):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        backbone (str): model type of backbone
        block (list): block type for layer4 or Res5
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
        initializer (dict): config for module parameters initialization
    """

    def __init__(self,
                 inplanes,
                 backbone,
                 num_classes,
                 cfg,
                 deformable=False,
                 block=None,
                 bn={FREEZE: True},
                 initializer=None):
        super(Res5, self).__init__(inplanes, num_classes, cfg)

        setup_bn(bn)

        from unn.models.backbones.resnet import BasicBlock
        from unn.models.backbones.resnet import Bottleneck
        from unn.models.backbones.resnet import DeformBasicBlock
        from unn.models.backbones.resnet import DeformBlock
        from unn.models.backbones.resnet import make_layer4

        if block is None:
            if backbone in ['resnet18', 'resnet34']:
                if deformable:
                    block = DeformBasicBlock
                else:
                    block = BasicBlock
            elif backbone in ['resnet50', 'resnet101', 'resnet152']:
                if deformable:
                    block = DeformBlock
                else:
                    block = Bottleneck
            else:
                raise NotImplementedError(f'{backbone} is not supported for Res5 head')
        else:
            logger.warning(
                "Argument `block` will be deprecated soon, which is now infered by `backbone` and `deformable`")
            nets = {
                ('resnet18', 'resnet34'): ['BasicBlock', 'DeformBasicBlock'],
                ('resnet50', 'resnet101', 'resnet152'): ['Bottleneck', 'DeformBlock']
            }
            for bbe, blk in nets.items():
                if backbone in bbe:
                    assert block in blk, f'for {bbe}, optional blocks are {blk}'
            block = {
                'Bottleneck': Bottleneck,
                'DeformBlock': DeformBlock,
                'BasicBlock': BasicBlock,
                'DeformBasicBlock': DeformBasicBlock
            }[block]

        layer = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3]
        }[backbone]

        stride = cfg['roipooling']['pool_size'] // 7
        self.layer4 = make_layer4(self.inplanes, block, 512, layer[3], stride=stride)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc_cls = nn.Linear(512 * block.expansion, num_classes)
        if self.cfg.get('share_location', False):
            self.fc_loc = nn.Linear(512 * block.expansion, 4)
        else:
            self.fc_loc = nn.Linear(512 * block.expansion, num_classes * 4)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_cls, 0.01)
        init_weights_normal(self.fc_loc, 0.001)

        rollback_bn()

    def predict(self, rois, x, stride):
        x = self.roipool(rois, x, stride)
        x = self.layer4(x)
        x = self.avgpool(x)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        cls_pred = self.fc_cls(x)
        loc_pred = self.fc_loc(x)
        return cls_pred, loc_pred


class RFCN(BboxNet):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        feat_planes (int): channels of intermediate features
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
        initializer (dict): config for module parameters initialization
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        super(RFCN, self).__init__(inplanes, num_classes, cfg)

        ps = self.pool_size
        inplanes = self.inplanes

        self.new_conv = nn.Conv2d(inplanes, feat_planes, kernel_size=1, bias=False)
        self.rfcn_score = nn.Conv2d(feat_planes, ps * ps * num_classes, kernel_size=1)
        if self.cfg.get('share_location', False):
            self.rfcn_bbox = nn.Conv2d(feat_planes, ps * ps * 4, kernel_size=1)
        else:
            self.rfcn_bbox = nn.Conv2d(feat_planes, ps * ps * 4 * num_classes, kernel_size=1)
        self.pool = nn.AvgPool2d((ps, ps), stride=(ps, ps))

        initialize_from_cfg(self, initializer)

    def predict(self, rois, x, stride):
        x = self.new_conv(x)
        x_cls = self.rfcn_score(x)
        x_loc = self.rfcn_bbox(x)

        x_cls = self.roipool(rois, x_cls, stride)
        x_loc = self.roipool(rois, x_loc, stride)

        x_cls = self.pool(x_cls)
        x_loc = self.pool(x_loc)

        cls_pred = x_cls.squeeze(dim=-1).squeeze(dim=-1)
        loc_pred = x_loc.squeeze(dim=-1).squeeze(dim=-1)

        return cls_pred, loc_pred
