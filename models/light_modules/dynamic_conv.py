#!/usr/bin/env python
#Author: Daoming Zong
#Date: 2021-09-22 14:41:13
#LastEditTime: 2021-09-22 19:33:10
#LastEditors: Daoming Zong and Chunya Liu
#Description: 
#FilePath: /models/SmallT/models/light_modules/dynamic_conv.py
#Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   dynamic_conv.py
# @time    :   2021/09/22 14:41:31
# @authors  :  daoming zong, chunya liu
# @version :   1.0
# @contact :   zongdaoming@sensetime.com; liuchunya@sensetime.com
# @desc    :   None
# Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dynamic Convolution module."""

import numpy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import linear, pad, softmax, dropout

MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class DynamicConvolution(nn.Module):
    """Dynamic Convolution layer.
    This implementation is based on
    https://github.com/pytorch/fairseq/tree/master/fairseq
    Args:
        wshare (int): the number of kernel of convolution
        n_feat (int): the number of features
        dropout_rate (float): dropout_rate
        kernel_size_str (str): kernel size (length)
        lnum (inst): index of layer
        use_kernel_mask (bool): Use causal mask or not for convolution kernel
        use_bias (bool): Use bias term or not.
    """
    def __init__(
        self,
        wshare,
        n_feat,
        dropout_rate,
        kernel_size_str,
        lnum,
        use_kernel_mask=False,
        use_bias=False,
    ):
        """Construct Dynamic Convolution layer."""
        super(DynamicConvolution, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = int(kernel_size_str.split("_")[lnum])
        self.attn = None
        # linear -> GLU -- -> lightconv -> linear
        #               \        /
        #                 Linear
        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        # self.linear1 = nn.Linear(n_feat, n_feat)
        self.linear2 = nn.Linear(n_feat, n_feat)
        self.linear_weight = nn.Linear(n_feat, self.wshare * 1 * self.kernel_size)
        nn.init.xavier_uniform(self.linear_weight.weight)
        self.act = nn.GLU()

        # dynamic conv related
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))
                        
    def forward(self, query, key, value, mask, key_padding_mask):
        """Forward of 'Dynamic Convolution'.
        This function takes query, key and value but uses only quert.
        This is just for compatibility with self-attention layer (attention.py)
        Args:
            query (torch.Tensor): (batch, time1, d_model) input tensor
            key (torch.Tensor): (batch, time2, d_model) NOT USED
            value (torch.Tensor): (batch, time2, d_model) NOT USED
            mask (torch.Tensor): (batch, time1, time2) mask
        Return:
            x (torch.Tensor): (batch, time1, d_model) ouput
        """
        # linear -> GLU -- -> lightconv -> linear
        #               \        /
        #                 Linear
        x = query
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size

        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)

        # get kernel of convolution
        weight = self.linear_weight(x)  # B x T x kH
        weight = F.dropout(weight, self.dropout_rate, training=self.training)
        weight = weight.view(B, T, H, k).transpose(1, 2).contiguous()  # B x H x T x k
        weight_new = torch.zeros(B * H * T * (T + k - 1), dtype=weight.dtype)
        weight_new = weight_new.view(B, H, T, T + k - 1).fill_(float("-inf"))
        weight_new = weight_new.to(x.device)  # B x H x T x T+k-1
        weight_new.as_strided(
            (B, H, T, k), ((T + k - 1) * T * H, (T + k - 1) * T, T + k, 1)
        ).copy_(weight)
        weight_new = weight_new.narrow(-1, int((k - 1) / 2), T)  # B x H x T x T(k)
        if self.use_kernel_mask:
            kernel_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
            weight_new = weight_new.masked_fill(kernel_mask == 0.0, float("-inf"))
        weight_new = F.softmax(weight_new, dim=-1)
        self.attn = weight_new
        weight_new = weight_new.view(B * H, T, T)

        # convolution
        x = x.transpose(1, 2).contiguous()  # B x C x T
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)  # BH x T x C/H
        x = x.transpose(1, 2).contiguous().view(B, C, T)

        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)  # B x T x C

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        if key_padding_mask is not None and not self.use_kernel_mask:
           # import pdb; pdb.set_trace()
           # 运行结果: mask中数值为1的位置在 tensor a 中相应的位置填充成了0.00 
           #          mask中数值为0的位置在tensor a 中保留了原值  
            key_padding_mask = key_padding_mask.transpose(-1, -2).unsqueeze(2)
            x = x.masked_fill_(key_padding_mask, 0.0)
        # second linear layer
        x = self.linear2(x)
        return x
    
    