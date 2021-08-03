import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb

def PositionTransform(bbox1, bbox2, image_info):
    if not isinstance(bbox1, torch.FloatTensor):
        bbox1 = torch.HalfTensor(bbox1).cuda()
        bbox2 = torch.HalfTensor(bbox2).cuda()
        W = image_info[0][0].half()
        H = image_info[0][1].half()
    else:
        W = image_info[0][0]
        H = image_info[0][1]
    N = bbox1.shape[0]
    xmin1 = bbox1[:, 1] / H
    xmin2 = bbox2[:, 1] / H
    ymin1 = bbox1[:, 2] / W
    ymin2 = bbox2[:, 2] / W
    xmax1 = bbox1[:, 3] / H
    xmax2 = bbox2[:, 3] / H
    ymax1 = bbox1[:, 4] / W
    ymax2 = bbox2[:, 4] / W
    w1 = xmax1 - xmin1
    w2 = xmax2 - xmin2
    h1 = ymax1 - ymin1
    h2 = ymax2 - ymin2
    cx1 = (xmin1 + xmax1) / 2.0
    cy1 = (ymin1 + ymax1) / 2.0
    cx2 = (xmin2 + xmax2) / 2.0
    cy2 = (ymin2 + ymax2) / 2.0
    a1 = w1 * h1
    a2 = w2 * h2

    #delta_w = torch.log(w2 / torch.clamp(w1, min=0.0001)).view(N, 1)
    #delta_h = torch.log(h2 / torch.clamp(h1, min=0.0001)).view(N, 1)
    #delta_cx = torch.log(torch.abs(cx1 - cx2) / torch.clamp(w1, min=0.0001)).view(N, 1)
    #delta_cy = torch.log(torch.abs(cy1 - cy2) / torch.clamp(h1, min=0.0001)).view(N, 1)
    xmin1 = xmin1.view(N, 1)
    xmin2 = xmin2.view(N, 1)
    ymin1 = ymin1.view(N, 1)
    ymin2 = ymin2.view(N, 1)
    xmax1 = xmax1.view(N, 1)
    xmax2 = xmax2.view(N, 1)
    ymax1 = ymax1.view(N, 1)
    ymax2 = ymax2.view(N, 1)
    w1 = w1.view(N, 1)
    w2 = w2.view(N, 1)
    h1 = h1.view(N, 1)
    h2 = h2.view(N, 1)
    a1 = a1.view(N, 1)
    a2 = a2.view(N, 1)
    
    position_mat = torch.cat((xmin1, xmin2, ymin1, ymin2, xmax1, xmax2, ymax1, ymax2, w1, w2, h1, h2, a1, a2), -1)
    
    return position_mat


def PositionEmbedding(position_mat, dim, wave_len=768):

    N = position_mat.shape[0]
    M = position_mat.shape[1]
    if not isinstance(position_mat, torch.FloatTensor):
        feat_range = torch.arange(dim / 2).cuda().half()
    else:
        feat_range = torch.arange(dim / 2)
    dim_mat = feat_range / (dim / 2)
    dim_mat = 1.0 / (torch.pow(wave_len, dim_mat))
    dim_mat = dim_mat.view(1,1,-1)

    position_mat = position_mat.view(N, M, 1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(N, -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1).view(N, -1)
    return embedding

