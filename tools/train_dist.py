#!/usr/bin/env python
# Copyright (c) CAIRI AI Lab. All rights reserved
"""
Distributed training script for PJLab GPU cluster.

This script supports:
- Single-node multi-GPU training
- Multi-node multi-GPU training
- Automatic environment detection

Usage:
    # Single node, 8 GPUs
    python tools/train_dist.py --config configs/mmnist_cifar/simvp/SimVP_gSTA.py --gpus 8

    # Multi-node (2 nodes, 16 GPUs) - run on master node
    python tools/train_dist.py --config configs/mmnist_cifar/simvp/SimVP_gSTA.py \
        --gpus 8 --nodes 2 --master_addr <MASTER_IP> --master_port 29500

    # On PJLab cluster with Slurm (see slurm_train_dist.sh)
    sbatch slurm_train_dist.sh
"""

import os
import os.path as osp
import sys
import warnings
import argparse
import time

warnings.filterwarnings('ignore')

import torch
from torch import distributed as dist
from argparse import Namespace

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, load_config,
                           update_config, get_dist_info)


def setup_dist_training(args):
    """Initialize distributed training environment."""

    # Check if already initialized by torchrun/srun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args['rank'] = int(os.environ['RANK'])
        args['world_size'] = int(os.environ['WORLD_SIZE'])
        args['local_rank'] = int(os.environ.get('LOCAL_RANK', args['local_rank']))
        args['dist'] = True
        print(f"Distributed training initialized: rank={args['rank']}, world_size={args['world_size']}")
    else:
        args['dist'] = False
        args['rank'] = 0
        args['world_size'] = 1
        args['local_rank'] = 0
        print("Single GPU or DataParallel mode")

    # Convert gpus from int to list for Lightning compatibility
    if isinstance(args.get('gpus'), int):
        args['gpus'] = list(range(args['gpus']))

    return args


def parse_args():
    # First, get base arguments from create_parser
    base_parser = create_parser()
    base_args = base_parser.parse_args([])
    base_dict = vars(base_args)

    # Add distributed training specific arguments
    parser = argparse.ArgumentParser(description='Distributed Training for OpenSTL')

    # Add base arguments with defaults from create_parser (exclude gpus, will add separately)
    for key, value in base_dict.items():
        if key not in ['config_file', 'gpus']:  # Exclude config_file and gpus
            # Handle boolean flags (action='store_true')
            if key in ['dist', 'fp16', 'torchscript', 'fps', 'test', 'deterministic',
                       'use_augment', 'use_prefetcher', 'drop_last', 'overwrite',
                       'no_display_method_info']:
                parser.add_argument(f'--{key}', action='store_true', default=value)
            else:
                parser.add_argument(f'--{key}', default=value, type=type(value) if value is not None else str)

    # Config file argument (different name to avoid conflict)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (overrides config_file)')

    # Distributed training arguments (override gpus from base_parser)
    parser.add_argument('--gpus', type=int, default=8,
                        help='Number of GPUs per node')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1',
                        help='Master node address for multi-node training')
    parser.add_argument('--master-port', type=str, default='29500',
                        help='Master port for multi-node training')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load config: prefer --config, fall back to --config_file
    config_path = args.config if args.config else args.config_file
    if config_path:
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
        # Build args dict
        cfg_dict = args.__dict__
        cfg_dict = update_config(cfg_dict, config, exclude_keys=[])
    else:
        cfg_dict = args.__dict__

    # Setup distributed training
    cfg_dict = setup_dist_training(cfg_dict)

    # Ensure epoch is an integer
    if isinstance(cfg_dict.get('epoch'), str):
        cfg_dict['epoch'] = int(cfg_dict['epoch'])

    # Convert dict back to Namespace object for BaseExperiment
    cfg = Namespace(**cfg_dict)

    # Set distributed environment variables for torchrun
    if cfg.dist:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(cfg.local_rank)
            os.environ['WORLD_SIZE'] = str(args.gpus * args.nodes)

    # Print training info
    rank, world_size = get_dist_info()
    if rank == 0:
        print('=' * 60)
        print('OpenSTL Distributed Training')
        print('=' * 60)
        print(f"Method: {cfg.method if hasattr(cfg, 'method') else 'N/A'}")
        print(f"Dataset: {cfg.dataname if hasattr(cfg, 'dataname') else 'N/A'}")
        print(f"GPUs per node: {args.gpus}")
        print(f"Number of nodes: {args.nodes}")
        print(f"Total GPUs: {world_size}")
        print(f"Batch size per GPU: {cfg.batch_size if hasattr(cfg, 'batch_size') else 'N/A'}")
        print(f"Effective batch size: {(cfg.batch_size if hasattr(cfg, 'batch_size') else 0) * world_size}")
        print('=' * 60)

    # Create and train
    exp = BaseExperiment(cfg, strategy='ddp')
    exp.train()

    if rank == 0:
        print('=' * 35 + ' testing ' + '=' * 35)
        exp.test()


if __name__ == '__main__':
    main()
