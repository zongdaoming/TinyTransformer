###
 # @Author: your name
 # @Date: 2021-08-22 23:11:07
 # @LastEditTime: 2021-08-26 09:56:46
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /models/SmallT/scripts/run_smca.sh
### 

#!/bin/sh
###
 # @Author: your name
 # @Date: 2021-08-22 23:02:34
 # @LastEditTime: 2021-08-22 23:10:58
 # @LastEditors: Please set LastEditors
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
    python -u -W ignore main_smca.py \
    --config $3 \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log

# sh ./scripts/run_smca.sh irdcRD 32 ./configs/small_transformer_ablation/smca.yaml
# sh ./scripts/run_smca.sh irdcRD 32 ./configs/small_transformer_ablation/smca_dec_layers.yaml
# sh ./scripts/run_smca.sh irdcRD 32 ./configs/small_transformer_ablation/smca_dim_feedward.yaml
# sh ./scripts/run_smca.sh irdcRD 24 ./configs/small_transformer_ablation/smca_dec_layers_dim_feedward.yaml
