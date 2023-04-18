#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

$PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py --dist \
    --config_file $CFG \
    --ex_name $WORK_DIR \
    --seed 42 --launcher="pytorch" ${PY_ARGS}
