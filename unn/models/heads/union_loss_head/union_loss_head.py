import logging
import copy
import json

import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from ..utils import loss as L

__all__ = ['UnionLossNet']
logger = logging.getLogger('global')

class UnionLossNet(nn.Module):
    def __init__(self, inplanes, cfg):
        super(UnionLossNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.prefix = 'UnionLossNet'
        
    def forward(self, input):
        output = {}
        if self.training:
            asso_predict = input['PairWiseNet.predict_vector'] 
            attr_predict = input['AttributeNet.predict_vector']
            asso_target = input['PairWiseNet.predict_target'] 
            attr_target = input['AttributeNet.predict_target'] 
            asso_score = self.find_margin_score(asso_predict, asso_target)
            attr_score = self.find_margin_score(attr_predict, attr_target)
            margin_score = min(asso_score, attr_score)
            asso_predict, asso_target = self.split(asso_predict, asso_target, margin_score)
            attr_predict, attr_target = self.split(attr_predict, attr_target, margin_score)
            loss_scale = self.cfg.get('loss_scale', 1.0)
            asso_loss = L.smooth_l1_loss(asso_predict.float(), asso_target.float(), sigma=1.0, normalizer=asso_predict.shape[0] * asso_predict.shape[1])
            attr_loss = L.smooth_l1_loss(attr_predict.float(), attr_target.float(), sigma=1.0, normalizer=attr_predict.shape[0] * attr_predict.shape[1])
            output[self.prefix + '.asso_loss'] = asso_loss * loss_scale
            output[self.prefix + '.attr_loss'] = attr_loss * loss_scale
        return output

    def find_margin_score(self, predict_vector, predict_target):
        score = 1.0
        N, C = predict_vector.shape
        for i in range(N):
            for j in range(C):
                if abs(predict_target[i][j] - 1) < 0.00001:
                    score = min(score, predict_vector[i][j].detach().cpu().numpy().tolist())
        return score

    def split(self, predict_vector, predict_target, margin_score):
        target = torch.zeros_like(predict_vector).cuda()
        mask = torch.zeros_like(predict_vector).cuda()
        N, C = predict_vector.shape
        for i in range(N):
            for j in range(C):
                if abs(predict_target[i][j] - 1) < 0.00001 or predict_vector[i][j] < margin_score:
                    mask[i][j] = 0
                else:
                    target[i][j] = margin_score
                    mask[i][j] = 1
        target = Variable(target)
        return predict_vector * mask, target
