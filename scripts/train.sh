#!/bin/sh
###
 # @Author: your name
 # @Date: 2021-08-22 23:02:34
 # @LastEditTime: 2021-09-07 13:50:37
 # @LastEditors: Daoming Zong and Chunya Liu
 # @Description: In User Settings Edit
 # @FilePath: /models/SmallT/scripts/train.sh
### 
# GPUS_PER_NODE=16
set -x 

PARTITION=$1
GPUS=$2
config=$3

declare -u expname
expname=`basename ${config} .yaml`

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

currenttime=`date "+%Y%m%d%H%M%S"`
g=$(($2<8?$2:8))

mkdir -p  results/${expname}/train_log

srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=${expname} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=$g \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u -W ignore main.py \
    --config $3 \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log

# sh ./scripts/train.sh irdcRD 24 ./configs/test.yaml
# sh ./scripts/train.sh test 8 ./configs/smca.yaml
# sh ./scripts/train.sh irdcRD 8 ./configs/small_transformer/levit_c_128s.yaml
# sh ./scripts/train.sh irdcRD 64 ./configs/small_transformer/levit_c_128s.yaml
# sh ./scripts/train.sh irdcRD 32 ./configs/small_transformer/levit_c_128s.yaml
# sh ./scripts/train.sh irdcRD 40 ./configs/small_transformer/levit_128.yaml
# sh ./scripts/train.sh test 8 ./configs/small_transformer/cross_detr.yaml
# sh ./scripts/train.sh irdcRD 40 ./configs/small_transformer/cross_detr.yaml
# sh ./scripts/train.sh irdcRD 72 ./configs/small_transformer/cross_detr.yaml
# sh ./scripts/train.sh test ./configs/small_transformer/cross_detr.yaml
# sh ./scripts/train.sh irdcRD 32 ./configs/small_transformer/cross_detr_flops_test.yaml

# sh ./scripts/train.sh irdcRD 48 ./configs/small_transformer/dpt_detr_flops_pretrained.yaml
# sh ./scripts/train.sh irdcRD 48 ./configs/small_transformer/dpt_detr_flops.yaml
# sh ./scripts/train.sh irdcRD 48 ./configs/small_transformer/dpt_detr_flops.yaml
# sh ./scripts/train.sh irdcRD 8 ./configs/small_transformer_ablation/detr_flops_dec_layers.yaml
# sh ./scripts/train.sh irdcRD 8 ./configs/small_transformer_ablation/detr_flops_dim_feedward.yaml
# sh ./scripts/train.sh irdcRD 8 ./configs/small_transformer_ablation/conditional_detr_flops.yaml
# sh ./scripts/train.sh test 1 ./configs/small_transformer_ablation/detr_flops.yaml
# sh ./scripts/train.sh irdcRD 4 ./configs/small_transformer_ablation/detr_flops_conv.yaml

# sh ./scripts/train.sh irdcRD 3 ./configs/dynamic_transformer/dynamic_transformer.yaml
# sh ./scripts/train.sh irdcRD 8 ./configs/dynamic_transformer/dynamic_transformer.yaml
