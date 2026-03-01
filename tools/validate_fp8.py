#!/usr/bin/env python
# Copyright (c) CAIRI AI Lab. All rights reserved
"""
FP8/FP16/BF16 Mixed Precision Training Validation Script

This script compares training performance between different precision modes:
- FP32 (baseline)
- FP16 (mixed precision)
- BF16 (mixed precision, requires Ampere or newer GPUs)
- FP8 (requires Hopper GPUs and transformer-engine)

Usage:
    python tools/validate_fp8.py --dataname moving_mnist --method simvp \
        --precision fp32 fp16 bf16 --epochs 10

Note:
    - FP8 requires NVIDIA Hopper GPUs (H100, H800) and transformer-engine
    - BF16 requires NVIDIA Ampere or newer (A100, A800, RTX30xx, etc.)
    - FP16 works on most modern GPUs with Tensor Cores
"""

import os
import os.path as osp
import argparse
import time
import json
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Validate mixed precision training')
    parser.add_argument('--dataname', type=str, default='moving_mnist',
                        help='Dataset name')
    parser.add_argument('--method', type=str, default='simvp',
                        help='Method name')
    parser.add_argument('--precision', nargs='+', default=['fp32', 'fp16'],
                        choices=['fp32', 'fp16', 'bf16', 'fp8'],
                        help='Precision modes to test')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for validation')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='GPU IDs')
    parser.add_argument('--save-dir', type=str, default='work_dirs/fp8_validation',
                        help='Directory to save results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print configuration without running')
    args = parser.parse_args()
    return args


def check_precision_support(precision_mode):
    """Check if the GPU supports the specified precision mode."""
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    compute_capability = torch.cuda.get_device_capability(device)

    support_info = {
        'fp32': True,
        'fp16': compute_capability[0] >= 7,  # Volta or newer
        'bf16': compute_capability[0] >= 8,  # Ampere or newer
        'fp8': compute_capability[0] >= 9,   # Hopper
    }

    return support_info.get(precision_mode, False), device_name, compute_capability


def run_precision_test(args, precision_mode):
    """Run a single precision test and return metrics."""
    from openstl.api import BaseExperiment
    from openstl.utils import Config

    # Create config
    cfg = Config(
        dataname=args.dataname,
        method=args.method,
        epoch=args.epochs,
        batch_size=args.batch_size,
        gpus=args.gpus,
        precision=precision_mode,
        seed=42,
        ex_name=f'{args.method}_{args.dataname}_{precision_mode}',
        res_dir=args.save_dir,
        log_step=1,
        metric_for_bestckpt='val_loss',
        sched='cosine',
        lr=1e-3,
        warmup_epoch=2,
        min_lr=1e-5,
        opt='adam',
        weight_decay=0.0,
    )

    # Add dataset-specific defaults
    if args.dataname == 'moving_mnist':
        cfg.in_shape = (10, 1, 64, 64)
        cfg.total_length = 20
        cfg.aft_seq_length = 10
        cfg.pre_seq_length = 10

    print(f"\n{'='*60}")
    print(f"Testing precision: {precision_mode}")
    print(f"{'='*60}")

    # Check precision support
    supported, device_name, compute_cap = check_precision_support(precision_mode)
    print(f"GPU: {device_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"Precision supported: {supported}")

    if not supported:
        print(f"WARNING: {precision_mode} may not be fully supported on this GPU")
        if args.dry_run:
            return None

    start_time = time.time()

    try:
        exp = BaseExperiment(cfg)

        # Run training
        exp.train()

        train_time = time.time() - start_time

        # Get training metrics
        saved_dir = osp.join(exp.save_dir, 'saved')
        results_file = osp.join(saved_dir, 'results.json')

        if osp.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {'metrics': {}}

        return {
            'precision': precision_mode,
            'train_time': train_time,
            'train_time_per_epoch': train_time / args.epochs,
            'mae': results['metrics'].get('mae', None),
            'mse': results['metrics'].get('mse', None),
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'status': 'success'
        }

    except Exception as e:
        print(f"ERROR in {precision_mode} test: {str(e)}")
        return {
            'precision': precision_mode,
            'status': 'failed',
            'error': str(e)
        }
    finally:
        # Reset peak memory
        torch.cuda.reset_peak_memory_stats()


def main():
    args = parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*60)
    print("FP8/FP16/BF16 Mixed Precision Training Validation")
    print("="*60)
    print(f"Dataset: {args.dataname}")
    print(f"Method: {args.method}")
    print(f"Precision modes: {args.precision}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs: {args.gpus}")
    print(f"Save dir: {args.save_dir}")
    print("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return

    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Set GPU
    torch.cuda.set_device(args.gpus[0])

    # Run precision tests
    all_results = []
    for precision_mode in args.precision:
        result = run_precision_test(args, precision_mode)
        if result:
            all_results.append(result)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Precision':<12} {'Status':<10} {'Time/epoch':<15} {'MAE':<10} {'MSE':<10} {'Memory(MB)':<12}")
    print("-"*60)

    baseline_time = None
    for result in all_results:
        if result['status'] == 'success':
            time_per_epoch = f"{result['train_time_per_epoch']:.2f}s"
            mae = f"{result['mae']:.4f}" if result['mae'] else 'N/A'
            mse = f"{result['mse']:.4f}" if result['mse'] else 'N/A'
            memory = f"{result['peak_memory_mb']:.1f}"

            if result['precision'] == 'fp32':
                baseline_time = result['train_time_per_epoch']

            print(f"{result['precision']:<12} {result['status']:<10} {time_per_epoch:<15} {mae:<10} {mse:<10} {memory:<12}")
        else:
            print(f"{result['precision']:<12} FAILED: {result.get('error', 'Unknown error')}")

    # Calculate speedup
    if baseline_time:
        print("\n" + "="*60)
        print("Speedup (vs FP32)")
        print("="*60)
        for result in all_results:
            if result['status'] == 'success' and result['precision'] != 'fp32':
                speedup = baseline_time / result['train_time_per_epoch']
                print(f"{result['precision']}: {speedup:.2f}x faster")

    # Save results
    summary_file = osp.join(args.save_dir, 'fp8_validation_results.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {summary_file}")


if __name__ == '__main__':
    main()
