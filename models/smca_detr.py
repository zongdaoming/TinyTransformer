#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   smca_detr.py
# @time    :   2021/08/05 10:07:01
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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.misc_smca import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone_detr import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .smca_transformer import build_transformer
import numpy as np
#from fvcore.nn import sigmoid_focal_loss_jit
#from mmcv.ops import sigmoid_focal_loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class SMCADETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                 aux_loss=False, query_mixture=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_classes = num_classes 
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
#        pdb.set_trace()
        nn.init.constant_(self.class_embed.bias, bias_init_with_prob(0.01))
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.query_mixture = query_mixture

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
#        pdb.set_trace()
        if isinstance(samples[0], (list, torch.Tensor)):
            samples[0] =  (samples[0])
#        batch, channel, height, width = samples.decompose().shape
#        height, width = samples.decompose().shape[2], samples.decompose().shape[3]
        features, pos = self.backbone(samples[0])
#        h_w = samples[1][0]['size'].unsqueeze(0).unsqueeze(0)
        h_w = torch.stack([torch.stack([inst['size'] for inst in samples[1]])[:, 1], torch.stack([inst['size'] for inst in samples[1]])[:, 0]], dim=-1)
        h_w = h_w.unsqueeze(0)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs, points = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1], h_w)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        points = points[0, :, :, :].unsqueeze(0).repeat(6, 1, 1, 1)
        if self.query_mixture:
#           pdb.set_trace()
           outputs_coord[:, :, :, :2] = outputs_coord[:, :, :, :2] + points * torch.cat([torch.zeros(self.num_queries // 2), torch.ones(self.num_queries // 2)]).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(points)
        else:
           outputs_coord[:, :, :, :2] = outputs_coord[:, :, :, :2] + points
        outputs_coord = outputs_coord.sigmoid()
#        boxes1 = outputs_coord
#        pdb.set_trace()
#        if not (boxes1[:, :, :, 2:]>=boxes1[:, :, :, :2]).all():
#        if (boxes1[:, 2:] >= boxes1[:, :2]).all()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
#    pdb.set_trace()
    return loss.sum() / num_boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # empty_weight = torch.ones(self.num_classes + 1)
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # pdb.set_trace()
        # loss_ce = sigmoid_focal_loss(src_logits.reshape(-1, 92), target_classes.reshape(-1), 2, 0.25, None, 'none').sum() / sum([indice[0].shape[0] for indice in indices])
        # loss_ce = sigmoid_focal_loss(src_logits.reshape(-1, 92), target_classes.reshape(-1), 0.5, 0.25, None, 'mean')
        # src_logits_simoid = src_logits.sigmoid()       
        # alpha = 0.25
        # gamma = 2
        # target_classes_onehot = torch.FloatTensor(*src_logits.shape).to(src_logits.device)
        # target_classes_onehot.zero_()
        # target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        # pt = (1 - src_logits_simoid) * target_classes_onehot + src_logits_simoid * (1 - target_classes_onehot)
        # focal_weight = (alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)) * pt.pow(gamma)
        # loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction='none') * focal_weight
        # loss_ce = loss.sum() / (loss.shape[0] * loss.shape[1])
        # pdb.set_trace()
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # pdb.set_trace()
        # target_classes_onehot = torch.FloatTensor(*src_logits.shape).to(src_logits.device)
        # target_classes_onehot.zero_()
        # target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, sum([indice[0].shape[0] for indice in indices]))
        src_logits_orig = src_logits
        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        # pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        pos_inds = torch.nonzero(target_classes != 0, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        loss_ce = sigmoid_focal_loss_jit(
                src_logits,
                labels,
                0.25,
                2,
                reduction="sum",
            ) / num_boxes
        # target_classes_onehot = torch.FloatTensor(*src_logits.shape).to(src_logits.device)
        # target_classes_onehot.zero_()
        # target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        # target_classes_onehot[:, :, 91] = target_classes_onehot[:, :, 91] * self.eos_coef
        # prob = F.softmax(src_logits, dim=-1)
        # out = -(((1 - prob) ** 2) * F.log_softmax(src_logits, dim=-1) * target_classes_onehot)
        # prob = F.softmax(src_logits, dim=-1)
        # loss_ce = out.sum() / target_classes_onehot.sum()
        # pdb.set_trace()
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits_orig[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class PostProcessSMCA(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
#        pdb.set_trace()
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
    #     bs, query_len, category = out_logits.shape
    #     prob = out_logits.sigmoid()
    #     scores = prob.reshape(bs, -1)
    #     prob = F.softmax(out_logits, -1)
    #     scores, labels = prob[..., :-1].max(-1)
    #     bs, _, category = out_logits.shape
    # #      convert to [x0, y0, x1, y1] format
    #     boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    #     boxes = boxes.unsqueeze(2).repeat(1, 1, category, 1).reshape(bs, -1, 4
    # #      and from relative [0, 1] to absolute [0, height] coordinates
    #     img_h, img_w = target_sizes.unbind(1)
    #     scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    #     boxes = boxes * scale_fct[:, None, :]
    #     labels = torch.linspace(0, 91, steps=92).to(prob.device).unsqueeze(0).unsqueeze(0).repeat(bs, query_len, 1).reshape(bs, -1)        
    #     results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, query_len, category = out_logits.shape

        prob = out_logits.sigmoid()
        scores = prob.reshape(bs, -1)
#        scores, labels = prob[..., :-1].max(-1)
#        bs, _, category = out_logits.shape
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes.unsqueeze(2).repeat(1, 1, category, 1).reshape(bs, -1, 4)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        labels = torch.linspace(0, 91, steps=92).to(prob.device).unsqueeze(0).unsqueeze(0).repeat(bs, query_len, 1).reshape(bs, -1)
        results = []
        for i in range(bs):
#            pdb.set_trace()
            result = dict()
            scores_per_image, topk_indices = scores[i].topk(query_len, sorted=False)
            labels_per_image = labels[i][topk_indices]
            boxes_per_image = boxes[i][topk_indices]
            result['scores'] = scores_per_image
            result['labels'] = labels_per_image
            result['boxes'] = boxes_per_image
            results.append(result)
        return results

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # outputs['pred_boxes']'s output is ( center_x, center_y, width,  heighth)
        # out_logits.shape: torch.Size([3, 1500, 10])  out_bbox.shape: torch.Size([3, 1500, 4])
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        #prob.shape: torch.Size([3, 1500, 10])
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        # topk_boxes.shape: torch.Size([3, 100])
        topk_boxes = topk_indexes // out_logits.shape[2]
        # labels.shape: torch.Size([3, 100])
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = SMCADETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        query_mixture=args.query_mixture,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors