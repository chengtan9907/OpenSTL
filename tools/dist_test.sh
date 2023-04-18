#!/usr/bin/env bash

set -x

CFG=$1
GPUS=$2
CHECKPOINT=$3
PY_ARGS=${@:4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# test
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py --dist \
    --config_file $CFG \
    --ex_name $CHECKPOINT --launcher="pytorch" ${PY_ARGS}
