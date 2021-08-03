import copy
import json
import logging

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....extensions import nms
from ...initializer import init_bias_focal
from ...initializer import initialize_from_cfg
from ..utils import accuracy as A  # noqa F401
from ..utils import loss as L
from ..utils.anchor_generator import get_anchors_over_grid
from ..utils.anchor_generator import get_anchors_over_plane
from .anchor import anchor_targets
from .anchor import predict_rois


__all__ = ['NaiveRPN', 'RetinaSubNet']

logger = logging.getLogger('global')


class RoINet(nn.Module):
    """
    Head for the first stage detection task
    0 is always for the background class.

    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        num_classes (int): number of classes, including the background class, RPN is 2, RetinaNet is 81
        cfg (dict): config for training or test
    """

    def __init__(self, inplanes, num_classes, cfg):
        super(RoINet, self).__init__()
        self.origin_cfg = copy.deepcopy(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.tocaffe = self.cfg.get('tocaffe', False)
        self.num_classes = num_classes
        assert self.num_classes > 1
        self.num_anchors = len(cfg['anchor_ratios']) * len(cfg['anchor_scales'])
        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

    def get_anchor_targets(self, *args, **kwargs):
        raise NotImplementedError

    def get_cls_loss_type(self):
        raise NotImplementedError

    def get_loss(self, cls_pred, cls_target, cls_mask, loc_pred, loc_target, loc_mask):
        """
        Arguments:
            cls_pred (FloatTensor, fp32): [B, K, C]
            cls_target (LongTensor): [B, K], {-1, 0, 1} for RPN, {-1, 0, 1...C} for retinanet
            cls_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0
            loc_pred (FloatTensor, fp32): [B, K, 4]
            loc_target (FloatTensor, fp32): [B, K, 4]
            loc_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0  # only positive samples

        Returns:
            cls_loss, loc_loss, acc (FloatTensor, fp32)
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Note: the return type must be fp32 for fp16 support !!!

        Arguments:
            x: [B, inplanes, h, w], input feature
        Returns:
            cls_pred (FloatTensor), [B, num_anchors*num_classes, h, w]
            loc_pred (FloatTensor), [B, num_anchors*4, h, w]
        """
        raise NotImplementedError

    def generate_anchors(self, preds, strides, cfg):
        """
        Generate anchors for each feature map.

        Note:Numerical type of generated anchors is the same as cls_pred

        Arguments:
            preds (list of tuple): [L, (cls_pred, loc_pred)]
            strides (list): stride of different layers
            cfg (dict): config for generating anchors

        Returns:
            mlvl_anchors (list of Tensors)
        """
        mlvl_anchors = []
        for lvl_idx, (cls_pred, loc_pred) in enumerate(preds):
            h, w = loc_pred.shape[-2:]
            stride = strides[lvl_idx]
            # [K, 4]  K=h*w*A, A=len(anchor_ratios)*len(anchor_scales)
            anchors = get_anchors_over_plane(
                h,
                w,
                stride,
                cfg['anchor_ratios'],
                cfg['anchor_scales'],
                stride,
                dtype=cls_pred.dtype,
                device=cls_pred.device)
            mlvl_anchors.append(anchors)
        return mlvl_anchors

    def repermute_preds(self, preds):
        """Permute preds to [B, h*w*A, :]"""
        mlvl_cls_pred, mlvl_loc_pred, shapes = [], [], []
        for lvl_idx, (cls_pred, loc_pred) in enumerate(preds):
            b, a4, h, w = loc_pred.shape
            # [B,A*x,h,w]->[B, h*w*A, x]
            k = a4 * h * w // 4
            assert a4 % 4 == 0
            mlvl_cls_pred.append(cls_pred.permute(0, 2, 3, 1).contiguous().view(b, k, -1))
            mlvl_loc_pred.append(loc_pred.permute(0, 2, 3, 1).contiguous().view(b, k, 4))
            shapes.append((h, w, k))
        return mlvl_cls_pred, mlvl_loc_pred, shapes

    def get_cls_score(self, cls_pred):
        """The last dim should be foreground classes if loss_type is softmax"""
        if self.get_cls_loss_type() == 'softmax':
            # exclude the background class
            return F.softmax(cls_pred, dim=-1)[..., 1:]
        return cls_pred.sigmoid()

    def merge_mlvl_rois(self, mlvl_rois, cfg):
        """
        Merge rois from different levels together

        Note:
            do nms per image only for RetinaNet

        Arguments:
            mlvl_rois (FloatTensor): [N, >=7], (batch_index, x1, y1, x2, y2, score, cls)
            1. do nms for each class when nms_iou_thresh > 0
            2. keep top_n rois for each image, keep all when top_n <= 0
        """
        cfg_nms = cfg.get('nms', {'type': 'naive', 'nms_iou_thresh': -1})
        top_n = cfg.get('top_n', -1)

        merged_rois = []
        B = int(torch.max(mlvl_rois[:, 0]).item() + 1)
        C = self.num_classes
        for b_ix in range(B):
            _rois = mlvl_rois[mlvl_rois[:, 0] == b_ix]
            if _rois.numel() == 0: continue  # noqa E701

            if cfg_nms['nms_iou_thresh'] > 0:
                # for RetinaNet nms_iou_thresh is positive, otherwise -1
                rois = []
                for cls in range(1, C):
                    _cls_rois = _rois[_rois[:, 6] == cls]
                    if _cls_rois.numel() == 0: continue  # noqa E701
                    _, order = _cls_rois[:, 5].sort(descending=True)
                    _cls_rois = _cls_rois[order]
                    _, index = nms(_cls_rois[:, 1:6], cfg_nms)
                    _cls_rois = _cls_rois[index]
                    rois.append(_cls_rois)
                if len(rois) == 0: continue  # noqa E701
                rois = torch.cat(rois, dim=0)
            else:
                rois = _rois

            if top_n > 0 and top_n < rois.shape[0]:
                _, inds = torch.topk(rois[:, 5], top_n)
                rois = rois[inds]
            merged_rois.append(rois)
        if len(merged_rois) > 0:
            merged_rois = torch.cat(merged_rois, dim=0)
        else:
            merged_rois = mlvl_rois.new_zeros((1, 7))
        return merged_rois

    def forward(self, input):
        """
        Arguments:
            input['features'] (list): input feature layers, for C4 from backbone, others from FPN
            input['strides'] (list): strides of input feature layers
            input['image_info'] (list of FloatTensor): [B, 5] (resized_h, resized_w, scale_factor, origin_h, origin_w)
            input['gt_bboxes'] (list of FloatTensor): [B, 5] (x1, y1, x2, y2, label)
            input['gt_ignores'] (list of FloatTensor): [B, 4] (x1, y1, x2, y2), optional

        Returns:
            output['dt_bboxes'] (FloatTensor): predicted proposals
        """
        prefix = 'RoINet'
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        self.cfg = copy.deepcopy(self.origin_cfg)
        mode = input.get('runner_mode', 'val')
        if mode in self.origin_cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))
        output = {}
        mlvl_preds = [self.predict(x) for x in features]
        if self.tocaffe:
            aligned = self.align2caffe(mlvl_preds, self.get_cls_loss_type())
            for i in range(len(aligned)):
                output[prefix + '.blobs.classification' + str(i)] = aligned[i][0]
                output[prefix + '.blobs.localization' + str(i)] = aligned[i][1]
            self.output_anchor(self.cfg['anchor_ratios'], self.cfg['anchor_scales'], strides)

        # [hi*wi*A, 4], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_anchors = self.generate_anchors(mlvl_preds, strides, self.cfg)

        # [B, hi*wi*A, :]
        mlvl_cls_pred, mlvl_loc_pred, mlvl_shapes = self.repermute_preds(mlvl_preds)

        if self.training:
            gt_bboxes = input['gt_bboxes']
            ignore_regions = input.get('gt_ignores', None)
            all_anchors = torch.cat(mlvl_anchors, dim=0)
            cls_pred = torch.cat(mlvl_cls_pred, dim=1)
            loc_pred = torch.cat(mlvl_loc_pred, dim=1)

            cls_target, loc_target, cls_mask, loc_mask = self.get_anchor_targets(all_anchors, gt_bboxes, image_info, self.cfg, ignore_regions)
            cls_loss, loc_loss, acc = self.get_loss(cls_pred, cls_target, cls_mask, loc_pred, loc_target, loc_mask, mlvl_shapes)

            output[prefix + '.cls_loss'] = cls_loss * self.cfg.get('cls_loss_scale', 1.0)
            output[prefix + '.loc_loss'] = loc_loss * self.cfg.get('loc_loss_scale', 1.0)
            output[prefix + '.accuracy'] = acc

        # if training, for RetinaNet, forward is end; for faster rcnn, continue to predict proposals
        if self.is_end and self.training:
            return output

        # predict proposals layer by layer
        mlvl_rois = []
        for lvl_idx, anchors in enumerate(mlvl_anchors):
            cls_pred = mlvl_cls_pred[lvl_idx]
            loc_pred = mlvl_loc_pred[lvl_idx]
            cls_pred = self.get_cls_score(cls_pred)  # for softmax, excluding the background class

            rois = predict_rois(anchors, cls_pred.detach(), loc_pred.detach(), image_info, self.cfg)
            if rois.numel() > 0:
                mlvl_rois.append(rois)

        if len(mlvl_rois) > 0:
            rois = torch.cat(mlvl_rois, dim=0)
            if len(mlvl_anchors) > 1:
                # for FPN, merge rois from multi layers
                rois = self.merge_mlvl_rois(rois, self.cfg['across_levels'])
        else:
            rois = features.new_zeros((1, 7))
        output['dt_bboxes'] = rois
        return output

    def align2caffe(self, mlvl_preds, cls_loss_type):
        aligned_preds = []
        for cls_pred, loc_pred in mlvl_preds:
            if cls_loss_type == 'sigmoid':
                cls_pred = cls_pred.sigmoid()
            else:
                b, a4, h, w = loc_pred.shape
                a = a4 // 4
                c = cls_pred.shape[1] // a
                cls_pred = F.softmax(cls_pred.view(-1, c, h, w), dim=1).view(-1, a * c, h, w)
            aligned_preds.append((cls_pred, loc_pred))
        return aligned_preds

    def output_anchor(self, ratios, scales, strides):
        output = {'ratios': ratios, 'scales': scales, 'strides': strides, 'anchors': {}}
        for stride in strides:
            anchors = get_anchors_over_grid(ratios, scales, stride)
            output['anchors'][stride] = anchors.tolist()
        json.dump(output, open('anchors.json', 'w'), indent=2)


class NaiveRPN(RoINet):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        feat_planes (int): channels of intermediate features
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
        initializer (dict): config for module parameters initialization
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        super(NaiveRPN, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.conv3x3 = nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1)
        self.relu3x3 = nn.ReLU(inplace=True)
        self.cls_loss_type = self.cfg['cls_loss']

        C = {'sigmoid': -1, 'softmax': 0}[self.cls_loss_type] + self.num_classes
        self.conv_cls = nn.Conv2d(feat_planes, self.num_anchors * C, kernel_size=1, stride=1)
        self.conv_loc = nn.Conv2d(feat_planes, self.num_anchors * 4, kernel_size=1, stride=1)
        self.is_end = False

        initialize_from_cfg(self, initializer)
        self.freeze = self.cfg.get('freeze', False)
        if self.freeze:
            for name, module in self.named_children():
                for param in module.parameters():
                    param.requires_grad = False

    def get_anchor_targets(self, *args, **kwargs):
        cls_target, loc_target, cls_mask, loc_mask = anchor_targets(*args, **kwargs)
        cls_target = torch.clamp(cls_target, max=1)  # for naive rpn, only foreground and background classes
        return cls_target, loc_target, cls_mask, loc_mask

    def get_cls_loss_type(self):
        return self.cls_loss_type

    def get_loss(self, cls_pred, cls_target, cls_mask, loc_pred, loc_target, loc_mask, mlvl_shapes=None):

        cls_loss_type = self.get_cls_loss_type()
        sigma = self.cfg.get('smooth_l1_sigma', 3)
        normalizer = max(1, torch.sum(cls_mask).item())

        loc_loss = L.get_rpn_loc_loss(loc_pred, loc_target, loc_mask, sigma, normalizer)
        if self.cfg.get('focal_loss', None) is None:
            cls_loss, acc = L.get_rpn_cls_loss(cls_pred, cls_target, cls_mask, cls_loss_type)
        else:
            cls_loss, acc = L.get_focal_loss(cls_pred, cls_target, normalizer, self.num_classes, self.cfg['focal_loss'])
        return cls_loss, loc_loss, acc

    def predict(self, x):
        x = self.conv3x3(x)
        x = self.relu3x3(x)
        cls_pred = self.conv_cls(x)
        loc_pred = self.conv_loc(x)
        return cls_pred.float(), loc_pred.float()  # the return type must be fp32 for fp16 support !!!


class RetinaSubNet(RoINet):
    """
    Arguments:
        inplanes (list or int): input channel, which is a number or list contains a single element
        feat_planes (int): channels of intermediate features
        num_classes (int): number of classes, including the background class
        cfg (dict): config for training or test
        initializer (dict): config for module parameters initialization
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        super(RetinaSubNet, self).__init__(inplanes, num_classes, cfg)

        self.cls_subnet = nn.Sequential(
            nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_planes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_planes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_planes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.box_subnet = nn.Sequential(
            nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_planes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_planes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_planes, feat_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cls_loss_type = self.cfg['focal_loss']['type']
        init_prior = self.cfg['focal_loss']['init_prior']
        C = {'sigmoid': -1, 'softmax': 0}[self.cls_loss_type] + self.num_classes

        self.cls_subnet_pred = nn.Conv2d(feat_planes, self.num_anchors * C, kernel_size=3, stride=1, padding=1)
        self.box_subnet_pred = nn.Conv2d(feat_planes, self.num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.is_end = True

        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.cls_loss_type, init_prior, self.num_classes)

        if cfg.get('ghmc_loss', None) is not None:
            self.ghmc_loss = L.GHMCLoss.from_params(cfg['ghmc_loss'])
        if cfg.get('ghmr_loss', None) is not None:
            self.ghmr_loss = L.GHMRLoss.from_params(cfg['ghmr_loss'])

    def get_anchor_targets(self, *args, **kwargs):
        cls_target, loc_target, cls_mask, loc_mask = anchor_targets(*args, **kwargs)
        return cls_target, loc_target, cls_mask, loc_mask

    def get_cls_loss_type(self):
        return self.cls_loss_type

    def get_loss(self, cls_pred, cls_target, cls_mask, loc_pred, loc_target, loc_mask, mlvl_shapes=None):

        normalizer = max(1, torch.sum(loc_mask).item())
        # cls loss
        if self.cfg.get('ghmc_loss', None) is not None:
            assert self.get_cls_loss_type() == 'sigmoid'
            cls_loss = self.ghmc_loss(cls_pred, cls_target, mlvl_shapes)
            acc = torch.cuda.FloatTensor([0])
            # acc = A.accuracy(cls_pred, cls_target.long()-1, ignore_indices=[-1,-2])[0]
        else:
            cls_loss, acc = L.get_focal_loss(cls_pred, cls_target, normalizer, self.num_classes, self.cfg['focal_loss'])

        # loc loss
        if self.cfg.get('ghmr_loss', None) is not None:
            loc_loss = self.ghmr_loss(loc_pred, loc_target, loc_mask, mlvl_shapes)
        else:
            sigma = self.cfg.get('smooth_l1_sigma', 3)
            loc_loss = L.get_rpn_loc_loss(loc_pred, loc_target, loc_mask, sigma, normalizer)
        return cls_loss, loc_loss, acc

    def predict(self, x):
        cls_feature = self.cls_subnet(x)
        box_feature = self.box_subnet(x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        return cls_pred.float(), loc_pred.float()  # the return type must be fp32 for fp16 support !!!
