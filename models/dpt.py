#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   dpt.py
# @time    :   2021/08/16 09:29:23
# @authors  :  daoming zong, chunya liu
# @version :   1.0
# @contact :   zongdaoming@sensetime.com; liuchunya@sensetime.com
# @desc    :   Main Reference from https://github.com/CASIA-IVA-Lab/DPT
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
import copy
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
import torch.nn.functional as F
from typing import Optional, List
from collections import OrderedDict

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from utils.log_helper import default_logger as logger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .box_coder import *
from .depatch_embed import Simple_DePatch
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint

#__all__ = [
#    'depvt_tiny'#, 'pvt_small', 'pvt_medium', 'pvt_large'
#]

MODEL_REGISTRY = {}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class DeformablePatchTransformer(nn.Module):
    def __init__(
                 self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1], 
                 F4=True, 
                 patch_embeds=None,
                 pretrained=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4

        # patch_embed
        self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4 = patch_embeds

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches + 1, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        self.apply(self._init_weights)
        self.init_weights(pretrained)
        # new
        self.patch_embed2.reset_offset()
        self.patch_embed3.reset_offset()
        self.patch_embed4.reset_offset()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
            # Note that `mmcv` has a safe checkpoint_load impl, please refer to the following url
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)     
            # https://github.com/open-mmlab/mmcv/blob/571e3e5fc75c23b45cbd9b00011af094357c5f1d/mmcv/runner/checkpoint.py           
        """
        def _process_mmcls_checkpoint(checkpoint):
            # state_dict = checkpoint['state_dict']
            state_dict = checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
            new_checkpoint = dict(state_dict=new_state_dict)
            return new_checkpoint

        if isinstance(pretrained, str):
            # self.apply(self._init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            new_checkpoint = _process_mmcls_checkpoint(checkpoint)
            '''
            # ************************************************************************* Detailed code for loading checkpoint ********************************************************************************
            # Missing Keys: ['patch_embed1.proj.weight', 'patch_embed1.proj.bias', 'patch_embed1.norm.weight', 'patch_embed1.norm.bias', 'block1.0.norm1.weight', 'block1.0.norm1.bias', 'block1.0.attn.q.wei
            # ght', 'block1.0.attn.q.bias', 'block1.0.attn.kv.weight', 'block1.0.attn.kv.bias', 'block1.0.attn.proj.weight', 'block1.0.attn.proj.bias', 'block1.0.attn.sr.weight', 'block1.0.attn.sr.bias', '
            # block1.0.attn.norm.weight', 'block1.0.attn.norm.bias', 'block1.0.norm2.weight', 
            # Unexpected Keys: ['backbone.patch_embed1.proj.weight', 'backbone.patch_embed1.proj.bias', 'backbone.patch_embed1.norm.weight', 'backbone.patch_embed1.norm.bias', 'backbone.block1.0.norm1.weig
            # ht', 'backbone.block1.0.norm1.bias', 'backbone.block1.0.attn.q.weight', 'backbone.block1.0.attn.q.bias', 'backbone.block1.0.attn.kv.weight', 'backbone.block1.0.attn.kv.bias', 'backbone.block1
            # .0.attn.proj.weight', 'backbone.block1.0.attn.proj.bias', 'backbone.block1.0.attn.sr.weight', 'backbone.block1.0.attn.sr.bias', 'backbone.block1.0.attn.norm.weight', 'backbone.block1.0.attn.n
            # orm.bias', 'backbone.block1.0.norm2.weight',
            '''
            # pretrain_dict = new_checkpoint['state_dict']
            # my_model_dict = self.state_dict()
            # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in my_model_dict}
            # my_model_dict.update(pretrain_dict)
            # self.load_state_dict(my_model_dict)
            # print(f'load from {pretrained}.')
            logger.info(f"Loading pretrained model from {pretrained}")
            missing_keys, unexpected_keys = self.load_state_dict(new_checkpoint['state_dict'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                logger.info('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.info('Unexpected Keys: {}'.format(unexpected_keys))
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        pos_embed1 = self._get_pos_embed(self.pos_embed1, self.patch_embed1, H, W)
        x = x + pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        pos_embed2 = self._get_pos_embed(self.pos_embed2, self.patch_embed2, H, W)
        x = x + pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        
        pos_embed3 = self._get_pos_embed(self.pos_embed3, self.patch_embed3, H, W)
        x = x + pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        pos_embed4 = self._get_pos_embed(self.pos_embed4[:, 1:], self.patch_embed4, H, W)
        x = x + pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.F4:
            # x = x[3:4]
            x =x[-1]
        x = x.view(x.shape[0], x.shape[1],-1).permute(2, 0, 1)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def register_model_architecture(model_name):
    """
    New model architectures can be added to fairseq with the 
    func:`register_model_architecture` function decorator. 
    argument.
    For example::

        @register_model_architecture('lstm'')
        def lstm_luong_wmt_en_de(cfg):
            args.encoder_embed_dim = getattr(cfg.model, 'encoder_embed_dim', 1000)
            (...)
    The decorated function should take a single argument *cfg*, which is a
    :class:`omegaconf.DictConfig`. The decorated function should modify these
    arguments in-place to match the desired architecture.
    Args:
        model_name (str): the name of the Model 
    """
    def wrapper(fn):
        MODEL_REGISTRY[model_name] = fn
        return fn
    return wrapper

@register_model_architecture('dpt_tiny')
class dpt_tiny(DeformablePatchTransformer):
    def __init__(self, image_size = 224, pretrained=None, **kwargs):
        # patch_embed
        embed_dims=[64, 128, 320, 512]
        img_size =image_size
        Depatch = [False, True, True, True]
        patch_embeds=[]
        for i in range(4):
            inchans = embed_dims[i-1] if i>0 else 3
            in_size = img_size // 2**(i+1) if i>0 else img_size
            patch_size = 2 if i > 0 else 4
            if Depatch[i]:
                box_coder = pointwhCoder(input_size=in_size, patch_count=in_size//patch_size, weights=(1.,1.,1.,1.), pts=3, tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
                patch_embeds.append(
                    Simple_DePatch(box_coder, img_size=in_size, patch_size=patch_size, patch_pixel=3, patch_count=in_size//patch_size,
                    in_chans=inchans, embed_dim=embed_dims[i], another_linear=True, use_GE=True, with_norm=True))
            else:
                patch_embeds.append(
                    PatchEmbed(img_size=in_size, patch_size=patch_size, in_chans=inchans,
                            embed_dim=embed_dims[i]))
        super(dpt_tiny, self).__init__(
            img_size=img_size, 
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1], 
            drop_rate=0.0, 
            drop_path_rate=0.1, 
            patch_embeds=patch_embeds,
            F4 = True,
            pretrained=pretrained)

@register_model_architecture("dpn_small")
class dpt_small(DeformablePatchTransformer):
    def __init__(self, image_size = None, pretrained=None, **kwargs):
        # patch_embed
        embed_dims=[64, 128, 320, 512]
        img_size = image_size
        Depatch = [False, True, True, True]
        patch_embeds=[]
        for i in range(4):
            inchans = embed_dims[i-1] if i>0 else 3
            in_size = img_size // 2**(i+1) if i>0 else img_size
            patch_size = 2 if i > 0 else 4
            if Depatch[i]:
                box_coder = pointwhCoder(input_size=in_size, patch_count=in_size//patch_size, weights=(1.,1.,1.,1.), pts=3, tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
                patch_embeds.append(
                    Simple_DePatch(box_coder, img_size=in_size, patch_size=patch_size, patch_pixel=3, patch_count=in_size//patch_size,
                    in_chans=inchans, embed_dim=embed_dims[i], another_linear=True, use_GE=True, with_norm=True))
            else:
                patch_embeds.append(
                    PatchEmbed(img_size=in_size, patch_size=patch_size, in_chans=inchans,
                            embed_dim=embed_dims[i]))
        super(dpt_small, self).__init__(
            img_size = img_size, 
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], 
            drop_rate=0.0, 
            drop_path_rate=0.1, 
            patch_embeds=patch_embeds, 
            F4 = True,
            pretrained=pretrained)

@register_model_architecture("dpn_small_f4")
class dpt_small_f4(DeformablePatchTransformer):
    def __init__(self, image_size = None, pretrained=None,**kwargs):
        # patch_embed
        embed_dims=[64, 128, 320, 512]
        img_size = image_size
        Depatch = [False, True, True, True]
        patch_embeds=[]
        for i in range(4):
            inchans = embed_dims[i-1] if i>0 else 3
            in_size = img_size // 2**(i+1) if i>0 else img_size
            patch_size = 2 if i > 0 else 4
            if Depatch[i]:
                box_coder = pointwhCoder(input_size=in_size, patch_count=in_size//patch_size, weights=(1.,1.,1.,1.), pts=3, tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
                patch_embeds.append(
                    Simple_DePatch(box_coder, img_size=in_size, patch_size=patch_size, patch_pixel=3, patch_count=in_size//patch_size, 
                    in_chans=inchans, embed_dim=embed_dims[i], another_linear=True, use_GE=True, with_norm=True))
            else:
                patch_embeds.append(
                    PatchEmbed(img_size=in_size, patch_size=patch_size, in_chans=inchans,
                            embed_dim=embed_dims[i]))
        super(dpt_small_f4, self).__init__(
            img_size=img_size, 
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], 
            drop_rate=0.0, 
            drop_path_rate=0.1, 
            F4=True, 
            patch_embeds=patch_embeds, 
            pretrained=pretrained)

@register_model_architecture("dpt_medium")
class dpt_medium(DeformablePatchTransformer):
    def __init__(self, image_size = None, pretrained=None, **kwargs):
        # patch_embed
        embed_dims=[64, 128, 320, 512]
        img_size = image_size
        Depatch = [False, True, True, True]
        patch_embeds=[]
        for i in range(4):
            inchans = embed_dims[i-1] if i>0 else 3
            in_size = img_size // 2**(i+1) if i>0 else img_size
            patch_size = 2 if i > 0 else 12
            if Depatch[i]:
                box_coder = pointwhCoder(input_size=in_size, patch_count=in_size//patch_size, weights=(1.,1.,1.,1.), pts=3, tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
                patch_embeds.append(
                    Simple_DePatch(box_coder, img_size=in_size, patch_size=patch_size, patch_pixel=3, patch_count=in_size//patch_size,
                    in_chans=inchans, embed_dim=embed_dims[i], another_linear=True, use_GE=True, with_norm=True))
            else:
                patch_embeds.append(
                    PatchEmbed(img_size=in_size, patch_size=patch_size, in_chans=inchans, embed_dim=embed_dims[i]))
        
        super(dpt_medium, self).__init__(
            img_size=img_size, 
            patch_size=12,
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1], 
            drop_rate=0.0, 
            drop_path_rate=0.1, 
            patch_embeds=patch_embeds,
            F4 = True, 
            pretrained=pretrained)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of feedforward mode
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        """
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
        the embedding dimension.
        
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
        the embedding dimension.
        
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
        the embedding dimension.
        
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
        S is the source sequence length.         
        """ 
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class Transformer(nn.Module):
    def __init__(self,
                 encoder,
                 d_model=512, 
                 d_nhead=8, 
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu", 
                 normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.encoder = encoder
        decoder_layer = TransformerDecoderLayer(d_model, d_nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = d_nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src)
        
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None, query_pos=query_embed)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        # torch.Size([6, 400, 3, 768])
        return hs.transpose(1, 2)

def build_transformer(args):
    assert args.backbone in ["dpt_tiny", "dpt_small", "dpt_small_f4", "dpt_medium"], "Invalid backbone name: {}".format(args.backbone)
    encoder = MODEL_REGISTRY.get(args.backbone)(
        image_size=args.img_size, pretrained=args.pretrained)
    return Transformer(
        encoder = encoder, 
        d_model=args.dec_hidden_dim,
        dropout=args.dropout,
        d_nhead=args.dec_nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True)