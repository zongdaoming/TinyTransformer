#!/usr/bin/env python
#Author: Daoming Zong
#Date: 2021-09-22 14:39:16
#LastEditTime: 2021-09-22 19:46:40
#LastEditors: Daoming Zong and Chunya Liu
#Description: 
#FilePath: /models/SmallT/models/light_modules/lightconv.py
#Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.
#
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   lightconv.py
# @time    :   2021/09/22 14:39:21
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

"""Lightweight Convolution Module."""

import numpy
import torch
from torch import nn
import torch.nn.functional as F

MIN_VALUE = float(numpy.finfo(numpy.float32).min)

class LightweightConvolution(nn.Module):
    """Lightweight Convolution layer.
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
        """Construct Lightweight Convolution layer."""
        super(LightweightConvolution, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = int(kernel_size_str.split("_")[lnum])
        self.padding_size = int(self.kernel_size / 2)

        # linear -> GLU -> lightconv -> linear
        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat, n_feat)
        self.act = nn.GLU()

        # lightconv related
        self.weight = nn.Parameter(
            torch.Tensor(self.wshare, 1, self.kernel_size).uniform_(0, 1)
        )
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))

        # mask of kernel
        kernel_mask0 = torch.zeros(self.wshare, int(self.kernel_size / 2))
        kernel_mask1 = torch.ones(self.wshare, int(self.kernel_size / 2 + 1))
        self.kernel_mask = torch.cat((kernel_mask1, kernel_mask0), dim=-1).unsqueeze(1)

    def forward(self, query, key, value, mask, key_padding_mask):
        """Forward of 'Lightweight Convolution'.
        This function takes query, key and value but uses only query.
        This is just for compatibility with self-attention layer (attention.py)
        Args:
            query (torch.Tensor): (batch, time1, d_model) input tensor
            key (torch.Tensor): (batch, time2, d_model) NOT USED
            value (torch.Tensor): (batch, time2, d_model) NOT USED
            mask (torch.Tensor): (batch, time1, time2) mask
        Return:
            x (torch.Tensor): (batch, time1, d_model) ouput
        """
        # linear -> GLU -> lightconv -> linear
        x = query
        B, T, C = x.size()
        H = self.wshare

        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)

        # lightconv
        x = x.transpose(1, 2).contiguous().view(-1, H, T)  # B x C x T
        weight = F.dropout(self.weight, self.dropout_rate, training=self.training)
        if self.use_kernel_mask:
            self.kernel_mask = self.kernel_mask.to(x.device)
            weight = weight.masked_fill(self.kernel_mask == 0.0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        x = F.conv1d(x, weight, padding=self.padding_size, groups=self.wshare).view(
            B, C, T
        )
        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)  # B x T x C

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        if key_padding_mask is not None and not self.use_kernel_mask:
           # import pdb; pdb.set_trace()
           # ????????????: mask????????????1???????????? tensor a ??????????????????????????????0.00 
           #          mask????????????0????????????tensor a ??????????????????  
            key_padding_mask = key_padding_mask.transpose(-1, -2).unsqueeze(2)
            x = x.masked_fill_(key_padding_mask, 0.0)
            
        # second linear layer
        x = self.linear2(x)
        return x
