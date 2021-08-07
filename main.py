#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file    :   main_.py
# @time    :   2021/08/03 11:45:51
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
import os
import time
import yaml
import json
import random
import datetime
import argparse
from torch import cuda
from pathlib import Path

import torch
import numpy as np
import utils.misc_n as utils
from models import build_model
import datasets.samplers as samplers
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from utils import dynamic_load_trained_modules
from utils.log_helper import default_logger as logger
from timm.utils import get_state_dict, ModelEma

def main(args):
    # load config
    logger.info(f'Torch Version: {torch.__version__}')
    logger.info(f'Cuda Is Available: {torch.cuda.is_available()}')
    logger.info(f'Cuda number: {torch.cuda.device_count()}')
    logger.info(f'GPU Version: {torch.cuda.get_device_name()}')
    with open(args.config) as f:
        config = yaml.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    utils.init_distributed_mode(args)
    args.output_dir = os.path.join(
        'results', args.config.split('/')[-1].split('.')[0].upper())
    os.makedirs(args.output_dir, exist_ok=True)
    utils.print_args(args)
    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]    
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out    
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    # define optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # define dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)    
    # define dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val)
        else:
            # sampler_train = DistributedSampler(dataset_train)
            sampler_train = samplers.DistributedSampler(dataset_train)
            num_tasks = utils.get_world_size()
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')            
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)        
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=args.pin_mem)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=args.pin_mem)
    ############################################################ Load fintune weights ####################################################################
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        # print('load checkpoint from ' + args.checkpoint)
        # pretrain_dict = checkpoint['model']
        # # filter if needed
        # my_model_dict = model_without_ddp.state_dict()
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in my_model_dict}
        # my_model_dict.update(pretrain_dict)
        # model_without_ddp.load_state_dict(my_model_dict)
        
        # missing_keys, unexpected_keys = model_without_ddp.load_state_dict(pretrain_dict, strict=False)
        # unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        # if len(missing_keys) > 0:
        #     print('Missing Keys: {}'.format(missing_keys))
        # if len(unexpected_keys) > 0:
        #     print('Unexpected Keys: {}'.format(unexpected_keys))
        pretrain_dict = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias','head_dist.weight', 'head_dist.bias']:
            if k in pretrain_dict and pretrain_dict[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrain_dict[k]
        model.load_state_dict(pretrain_dict, strict=False)
    ############################################################ Load Checkpoint ##########################################################################
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
    ############################################################ Eval Mode ################################################################################
    if args.eval:
        test_stats, metric_res = evaluate(model, criterion, postprocessors,
                                              data_loader_val, args, device, args.output_dir)
        logger.info(**{f'test_{k}': v for k, v in test_stats.items()})
        logger.info(**metric_res)
        return
    ############################################################ Training  ################################################################################
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, 
        #                             loss_scaler, args.clip_grad, args.clip_mode, model_ema)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)                                    
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every args.save_freq epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)
        if (epoch + 1) % args.test_freq == 0:
            test_stats, metric_res = evaluate(model, criterion, postprocessors, data_loader_val, args, device, args.output_dir)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A smalll Transformer Backbone')
    parser.add_argument('--config', default='configs/coco_train.yaml', type=str, help='path to config file')
    args = parser.parse_args()
    main(args)