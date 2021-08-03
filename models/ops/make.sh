#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

#python setup.py build install
g=$(($2<8?$2:8))
T=`date +%m%d%H%M`
srun --mpi=pmi2 \
    --partition=$1 \
    --gres=gpu:$g -n 1 --ntasks-per-node=$g \
    --job-name=SETUP \
    python -u setup.py build install --user --record install_multiscale_abstraction.txt \
    2>&1 | tee log.install_multiscale_abstraction.$T
# sh make.sh irdcRD 1