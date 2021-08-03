import logging
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from ..utils import loss as L
from ..utils import accuracy as A

__all__ = ['GAP']

logger = logging.getLogger('global')

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().float().numpy() if torch.is_tensor(x) else x

class ClassificationNet(nn.Module):
    def __init__(self, inplanes, num_classes, cfg):
        super(ClassificationNet, self).__init__()
        self.origin_cfg = copy.deepcopy(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.cfg['num_classes'] = num_classes
        self.num_classes = num_classes
        self.inplanes = inplanes

    def predict(self, features):
        raise NotImplementedError

    def get_loss(self, input):

        features = input['features']
        gt_labels = input['gt_labels']
        pred_cls = self.predict(features)
        
        loss = {}
        acc = {}
        if self.cfg.get('cls_type', 'softmax') == 'softmax':
            cls_loss = F.cross_entropy(pred_cls.float(), gt_labels.long())
            accuracy = A.accuracy(pred_cls.float(), gt_labels.long())[0]
        else:
            cls_loss = F.binary_cross_entropy_with_logits(pred_cls, gt_labels)
        loss.update({'.cls_loss': cls_loss})
        acc.update({'.accuracy': accuracy})
        return loss, acc

    def get_labels(self, input):
        features = input['features']
        pred_cls = self.predict(features)
        if self.cfg.get('cls_type', 'softmax') == 'softmax':
            pred_cls = F.softmax(pred_cls, dim=1)
            B = pred_cls.shape[0]
            max_score, max_label = torch.max(pred_cls, dim=1)
            labels = torch.stack((max_label.float(), max_score.float()), dim=1)
        return labels

    def forward(self, input):
        prefix = 'ClassificationNet'
        mode = input.get('runner_mode', 'val')
        self.cfg = copy.deepcopy(self.origin_cfg)
        if mode in self.cfg:
            self.cfg.update(self.cfg[mode])
        else:
            self.cfg.update(self.cfg.get('val', {}))
        output = {}
        if self.training:
            loss, acc = self.get_loss(input)
            for k, v in loss.items():
                output[prefix + k] = v
            for k, v in acc.items():
                output[prefix + k] = v
        else:
            labels = self.get_labels(input)
            output['dt_labels'] = labels
        return output

class GAP(ClassificationNet):

    def __init__(self,
                 inplanes,
                 num_classes,
                 cfg,
                 initializer=None):
        super(GAP, self).__init__(inplanes, num_classes, cfg)
        inplanes = self.inplanes
        self.fc = nn.Linear(inplanes, num_classes)

    def predict(self, features):
        new_features = []
        for i in range(len(features)):
            feature = features[i]
            B = feature.shape[0]
            C = feature.shape[1]
            H = feature.shape[2]
            W = feature.shape[3]
            feature = feature.view(B * C * H, -1).contiguous()
            feature = torch.mean(feature, dim=1)
            feature = feature.view(B * C, -1).contiguous()
            new_features.append(torch.mean(feature, dim=1).view(B, C).contiguous())

        new_features = torch.stack(new_features)
        new_features = torch.mean(new_features, dim=0)
        new_features = self.fc(new_features)
        return new_features

