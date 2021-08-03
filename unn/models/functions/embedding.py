import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('global')

def ValueEmbedding(value, dim, wave_len=1000, dtype=torch.cuda.FloatTensor):

    origin_shape = value.shape
    N = value.numel()
    feat_range = torch.arange(dim / 2).type(dtype).cuda()
    dim_mat = feat_range / (dim / 2)
    dim_mat = 1.0 / (torch.pow(wave_len, dim_mat))
    dim_mat = dim_mat.view(1, -1)
    value_mat = 100. * value.view(N, 1)
    mul_mat = value_mat * dim_mat
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1).view(N, -1).contiguous()
    # dim, N
    embedding = embedding.permute(1, 0).contiguous()
    new_shape = torch.Size([int(dim)]) + origin_shape
    embedding = embedding.view(new_shape)
    return embedding

def ArrayEmbedding(array, dim, wave_len=768, dtype=torch.cuda.FloatTensor):
    N, C = array.shape
    embedding = []
    for c_ix in range(C):
        pos = ValueEmbedding(array[:, c_ix], dim, wave_len, dtype)
        pos = pos.permute(1, 0).contiguous()
        embedding.append(pos)

    embedding = torch.cat(embedding, dim=1)
    return embedding


def BoxPositionEmbedding(rois, h_num, w_num, dim, wave_len=1000, dtype=torch.cuda.FloatTensor):
    N = rois.shape[0]
    rois = torch.Tensor(rois).type(dtype).cuda()
    width = rois[:, 3] - rois[:, 1]
    height = rois[:, 4] - rois[:, 2]
    origin_x = rois[:, 1].view(N, 1)
    origin_y = rois[:, 2].view(N, 1)
    delta_x = width / w_num
    delta_y = height / h_num
    delta_x = delta_x.view(N, 1).contiguous()
    delta_y = delta_y.view(N, 1).contiguous()
    feat_range_x = torch.arange(w_num).type(dtype).cuda().view(1, -1).contiguous()
    feat_range_y = torch.arange(h_num).type(dtype).cuda().view(1, -1).contiguous()
    coordinate_x = origin_x + delta_x * feat_range_x
    coordinate_y = origin_y + delta_y * feat_range_y
    base = torch.ones((N, h_num, w_num)).type(dtype).cuda()
    coordinate_x = coordinate_x.view(N, 1, w_num) * base
    coordinate_y = coordinate_y.view(N, h_num, 1) * base
    embedding_x = ValueEmbedding(coordinate_x, dim / 2, wave_len, dtype)
    embedding_y = ValueEmbedding(coordinate_y, dim / 2, wave_len, dtype)
    embedding = torch.cat((embedding_x, embedding_y), dim=0)
    embedding = embedding.permute(1, 0, 2, 3).contiguous()
    return embedding


def ImagePositionEmbedding(b, c, h, w, dtype=torch.cuda.FloatTensor):
    rois = np.array([[0, 0, 0, h, w]])
    # C, 1, h, w
    embedding = BoxPositionEmbedding(rois, h, w, c, dtype=dtype)
    embedding = embedding.view(1, c, h, w)
    ones = torch.ones((b, c, h, w)).type(dtype).cuda()
    embedding = embedding * ones
    return embedding



