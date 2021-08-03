import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .... import extensions as E
from ...initializer import init_weights_normal
from ...initializer import initialize_from_cfg
from ..utils.assigner import map_rois_to_level
from .keypoint import keypoint_targets
from .keypoint import predict_keypoints

logger = logging.getLogger('global')

__all__ = ['ConvUp']


class KeypNet(nn.Module):
    """
    Generate keypoint predictions
    """
    def __init__(self, cfg):
        """
        Arguments:
            - cfg (:obj:`dict`):
        """
        super(KeypNet, self).__init__()

        self.origin_cfg = copy.deepcopy(cfg)  # keep the original configuration
        self.cfg = copy.deepcopy(cfg)  # runtime configuration
        self.tocaffe = self.cfg.get('tocaffe', False)

        if self.cfg.get('fpn', None):
            cfg_fpn = self.cfg['fpn']
            self.mlvl_roipool = E.MultiLevelGenericRoIPooling(self.cfg['roipooling'], cfg_fpn['fpn_levels'])
        else:
            self.roipool = E.build_generic_roipool(self.cfg['roipooling'])

    def predict(self, x):
        raise NotImplementedError

    def person_filter(self, bboxes):
        new_bboxes = []
        for i in range(bboxes.shape[0]):
            if abs(bboxes[i][4] - 1) < 0.00001:
                new_bboxes.append(bboxes[i])
        if len(new_bboxes) == 0:
            new_bboxes.append(torch.zeros((1, bboxes.shape[1])).type_as(bboxes).cuda())
        new_bboxes = torch.cat(new_bboxes, dim=0)
        return new_bboxes

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
                # (list of FloatTensor): [B] [num_gts, K, 3] (x, y, flag)
                'gt_keypoints': [],
                # (list of tuple): [B, (pos_inds, pos_target_gt_inds)],
                #  sampling results from bbox head
                'sample_record': [()]
            }

        Output example::

            # training output
            {'KeypNet.cls_loss': <tensor>}

            # testing output
            {
                # (FloatTensor), predicted boxes [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
                'dt_bboxes': <tensor>,
                # (FloatTensor) predicted keypoints [N, num_keyps, 3] (x, y, flag)
                'dt_keyps': <tensor>
            }
        """
        prefix = 'KeypNet'
        self.cfg = copy.deepcopy(self.origin_cfg)
        mode = input.get('runner_mode', 'val')
        if mode in self.cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))
        if self.training:
            return {prefix + '.keyp_loss': self.get_loss(input) * self.cfg.get('loss_scale', 1.0)}
        else:
            if input['dt_bboxes'].shape[0] == 0:
                return {}
            bboxes, keypoints, keyp_blobs = self.get_keyp(input)
            output = {'dt_bboxes': bboxes, 'dt_keyps': keypoints}
            if self.tocaffe:
                output[prefix + '.blobs.keypoint'] = keyp_blobs
            return output

    def get_logits(self, rois, features, strides, cfg):
        """
        Assign rois to each level and predict

        Note:
           1.The numerical type of `logits` must be fp32 for fp16 support !!!
           2.ONNX don't support indexing operation, so we do not recovering by indexing here

        Returns:
            rois (FloatTensor): assigned rois
            logits (FloatTensor): prediction of mask of assigned rois
            recover_inds (LongTensor): indices of recovering input rois from assigned rois
        """
        if cfg.get('fpn', None):
            fpn = cfg['fpn']
            if self.tocaffe and not self.training:
                # to save memory
                if rois.numel() > 0:
                    rois = rois[0:1]
                # make sure that all pathways included in the computation graph
                mlvl_rois, recover_inds = [rois] * len(fpn['fpn_levels']), None
            else:
                mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            pooled_feature = self.mlvl_roipool(mlvl_rois, features, strides)
            rois = torch.cat(mlvl_rois, dim=0)
            # pooled_feature = pooled_feature[recover_inds]
        else:
            pooled_feature = self.roipool(rois, features[0], strides[0])
            recover_inds = torch.arange(rois.shape[0], device=rois.device)
        logits = self.predict(pooled_feature)
        return rois, logits.float(), recover_inds  # `logits` must be fp32 for fp16 support !!!

    def get_loss(self, input):
        """
        Arguments:
            input (dict): data from last module
            input['features'] (list): input feature layers, for C4 from backbone, others from FPN
            input['strides'] (list): strides of input feature layers
            input['image_info'] (list of FloatTensor): [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
            input['dt_bboxes'] (FloatTensor): [B, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
            input['gt_bboxes'] (list of FloatTensor): [B, 5] (x1, y1, x2, y2, label)
            input['gt_keypoints'] (list of FloatTensor): [B, num_gts, K, 3] (x, y, flag)
            input['sample_record'] (list of tuple): [B, (pos_inds, pos_target_gt_inds)], sampling results from bbox head

        Returns:
            mask_loss (FloatTensor)
        """
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']
        gt_bboxes = input['gt_bboxes']
        gt_keypoints = input['gt_keyps']
        sample_record = input['sample_record']

        sampled_rois, keyps_target = keypoint_targets(sample_record, rois, gt_bboxes, gt_keypoints, image_info,
                                                      self.cfg)
        rois, keyp_logits, recover_inds = self.get_logits(sampled_rois, features, strides, self.cfg)
        keyp_logits = keyp_logits[recover_inds]

        R, K = keyp_logits.shape[:2]
        keyp_logits = keyp_logits.view(R * K, -1)
        keyps_target = keyps_target.view(-1)
        keyp_loss = F.cross_entropy(keyp_logits, keyps_target, ignore_index=-1)
        return keyp_loss

    def get_keyp(self, input):
        features = input['features']
        strides = input['strides']
        rois = input['dt_bboxes']

        rois, keyp_logits, recover_inds = self.get_logits(rois, features, strides, self.cfg)

        r, k, h, w = keyp_logits.shape
        keyp_logits = F.softmax(keyp_logits.view(-1, k, h * w), dim=2).view(-1, k, h, w)
        keypoints = predict_keypoints(rois, keyp_logits.detach())
        return rois, keypoints, keyp_logits


class ConvUp(KeypNet):
    """
    Use conv to generate keypoints prediction
    """
    def __init__(self, inplanes, num_keypoints, feat_planes, num_convs, deconv_kernel, cfg, initializer=None):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel, which is a number or list contains a single element
            - num_keypoints (:obj:`int`): 17 for coco dataset
            - feat_planes (:obj:`int`): channel of intermediate features
            - num_convs (:obj:`int`): number of convs before upsampling
            - deconv_kernel (:obj:`int`): kernel size of transposed conv
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization

        `Keypoints example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/keypoint-rcnn-R50-FPN-1x.yaml#L173-199>`_
        """
        super(ConvUp, self).__init__(cfg)

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        planes = feat_planes

        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)))
            inplanes = planes

        updeconv = nn.ConvTranspose2d(
            planes, num_keypoints, kernel_size=deconv_kernel, stride=2, padding=(deconv_kernel - 2) // 2)
        layers.append(updeconv)
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        self.head = nn.Sequential(*layers)
        initialize_from_cfg(self, initializer)
        init_weights_normal(updeconv, std=0.001)

    def predict(self, x):
        return self.head(x)
