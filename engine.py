#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   engine.py
# @time    :   2021/08/03 15:20:45
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
import math
import os
import sys
import datetime
import itertools 
import numpy as np
from torch import Tensor
from typing import Tuple
from typing import Iterable, Optional

import json
import torch
import torchvision
import utils.misc as utils
import torch.distributed as dist
from utils.misc import all_gather

from datasets.data_prefetcher_ml import data_prefetcher
from functools import reduce
from timm.data import Mixup
from collections import defaultdict
from utils.log_helper import default_logger as logger
from timm.utils import accuracy, ModelEma
from datasets.metrics.casino_evaluator import MREvaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 10
    print_freq = 100

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # clipping_value = 0.01 # arbitrary value of your choosing
            # grad_total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            # clipping_value = 0.01
            # grad_total_norm = utils.get_total_grad_norm(model.parameters(), clipping_value)
        
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_s(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None,
                    ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all gpus for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        import pdb
        pdb.set_trace()    

        optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()
        # if max_norm > 0:
        #     grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # else:
        #     grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(losses, optimizer, clip_grad=clip_grad, 
                    parameters=model.parameters(), 
                    create_graph=is_second_order)
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(grad_norm=grad_total_norm)
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold
    Returns:
        keep (Tensor): int64 tensor with the indices of
            the elements that have been kept by NMS, sorted
            in decreasing order of scores
    """
    if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    else:
        max_coordinate = boxes.max()
        # offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        offsets = idxs*(max_coordinate + torch.tensor(1).to(boxes.device))
        boxes_for_nms = boxes + offsets[:, None]
        # keep = nms(boxes_for_nms, scores, iou_threshold)
        # keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
        keep = torch.ops.torchvision.nms(boxes_for_nms, scores, iou_threshold)
        return keep

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, args, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    casino_evaluator = MREvaluator(
                                   gt_file=args.meta_file_test,
                                   num_classes=args.num_classes,
                                   class_names=args.class_names,
                                   iou_thresh=args.iou_thresh,
                                   metrics_csv=args.metrics_dir,
                                   )                                
    res_list_all_gather= []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device, non_blocking=True) if k!='image_id' else v for k, v in t.items()} for t in targets]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
 
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        #metric_logger.update(loss=0.0,loss2=0.0)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
      
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # Notice that  batch_sizes is also important
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        for target, output in zip(targets, results):
            """Non_max_suppression
            # Parameters:	
            # boxes (Tensor[N, 4]) – boxes where NMS will be performed. They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
            # scores (Tensor[N]) – scores for each one of the boxes
            # idxs (Tensor[N]) – indices of the categories for each one of the boxes.
            # iou_threshold (float) – discards all overlapping boxes with IoU > iou_threshold  
            # :return  int64 tensor with the indices of
            #          the elements that have been kept by NMS, sorted in decreasing order of scores
            """
            # perform non_max_suppression here
            filter_output = {}
            indices = batched_nms(output['boxes'], output['scores'], output['labels'], iou_threshold =0.8)
            filter_output['boxes'] = torch.index_select(output['boxes'],0,indices)
            filter_output['scores'] = torch.index_select(output['scores'],0,indices)
            filter_output['labels'] = torch.index_select(output['labels'],0,indices)
            res_list = []
            for idx in range(filter_output['boxes'].shape[0]):
                res_list.append(
                    {
                    'image_id':target['image_id'], 
                    'label': int(filter_output['labels'][idx]),
                    'bbox':  filter_output['boxes'][idx,:].tolist(),
                    'score': float(filter_output['scores'][idx])
                    })
            # accumulate predictions from all processes
            torch.distributed.barrier()                     
            res_list_all_gather.extend(list(itertools.chain.from_iterable(utils.all_gather(res_list))))                                       
            # No non_max_suppression settings
            # output = {k: v.detach().cpu().numpy() if k!='image_id' else v for k, v in output.items()}
            # res_list.append(output)
            # assert output['scores'].shape[0]  == output['labels'].shape[0] == output['boxes'].shape[0]
            # for JSON serializable prepare
            # for idx in range(output['boxes'].shape[0]):
            #     res_list.append(
            #         {
            #         'image_id':target[image_id'], 
            #         'label': int(outpu't['labels'][idx]),
            #         'bbox':  output['boxes'][idx,:].tolist(),
            #         'score': float(output['scores'][idx])
            #         })
            # res_list_all_gather.extend(list(itertools.chain.from_iterable(utils.all_gather(res_list))))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logger.info("Number of detection results: {}".format(len(res_list_all_gather)))
    metric_res = casino_evaluator.eval(res_list_all_gather)
    if args.res_dir is not None and utils.is_main_process():
        # results_file = os.path.join(args.res_dir, str(datetime.datetime.now())+'.json')  
        results_file = os.path.join(args.res_dir, datetime.datetime.now().strftime('%Y-%m-%d')+'.json')  
        with open(results_file, 'w') as writer:
            for idx_img, item in enumerate(res_list_all_gather):
                writer.write(json.dumps(item) + '\n')
        # with open(results_dir, 'w') as f:
        #     json.dump(res_list_all_gather, f)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, metric_res