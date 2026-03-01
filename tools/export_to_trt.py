#!/usr/bin/env python
# Copyright (c) CAIRI AI Lab. All rights reserved
"""
Export OpenSTL models to TensorRT for accelerated inference.

This script exports trained PyTorch models to TensorRT engine format
for faster inference on NVIDIA GPUs.

Usage:
    # Export SimVP model
    python tools/export_to_trt.py \\
        --config configs/mmnist_cifar/simvp/SimVP_gSTA.py \\
        --checkpoint work_dirs/simvp_mmnist/checkpoints/best.ckpt \\
        --save-dir work_dirs/trt_export \\
        --precision fp16 \\
        --batch-size 1 \\
        --seq-length 10 \\
        --height 64 \\
        --width 64

Requirements:
    - torch2trt: pip install torch2trt
    - tensorrt: pip install tensorrt
    - CUDA-compatible GPU with compute capability >= 6.0

Note:
    - FP16 requires compute capability >= 6.0 (Volta or newer)
    - FP8 requires compute capability >= 9.0 (Hopper)
"""

import os
import os.path as osp
import argparse
import logging
from pathlib import Path

import torch
import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not installed. Install with: pip install tensorrt")

try:
    from torch2trt import torch2trt
    TORCH2TRT_AVAILABLE = True
except ImportError:
    TORCH2TRT_AVAILABLE = False
    print("WARNING: torch2trt not installed. Install with: pip install torch2trt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Export OpenSTL model to TensorRT')

    # Model config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--method', type=str, default=None,
                        help='Method name (auto-detected from config if not specified)')

    # Export settings
    parser.add_argument('--save-dir', type=str, default='work_dirs/trt_export',
                        help='Directory to save exported model')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Precision mode for TensorRT')
    parser.add_argument('--max-batch-size', type=int, default=4,
                        help='Maximum batch size for TensorRT engine')

    # Input shape (can be overridden by config)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--seq-length', type=int, default=10,
                        help='Sequence length (pre_seq_length)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--height', type=int, default=64,
                        help='Input height')
    parser.add_argument('--width', type=int, default=64,
                        help='Input width')

    # torch2trt settings
    parser.add_argument('--max-workspace-size', type=int, default=1 << 30,
                        help='Max workspace size in bytes (default 1GB)')
    parser.add_argument('--strict-type-constraints', action='store_true',
                        help='Use strict type constraints')

    # Validation
    parser.add_argument('--validate', action='store_true',
                        help='Validate exported model against PyTorch')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for validation (absolute difference)')

    args = parser.parse_args()
    return args


def load_model_and_checkpoint(args):
    """Load model architecture and weights."""
    from openstl.methods import method_maps
    from openstl.utils import Config

    # Load config
    cfg = Config.fromfile(args.config)

    if args.method is None:
        args.method = cfg.get('method', 'simvp').lower()

    logger.info(f"Loading model: {args.method}")

    # Get input shape from config or args
    if hasattr(cfg, 'in_shape'):
        T, C, H, W = cfg.in_shape
        args.seq_length = T
        args.channels = C
        args.height = H
        args.width = W
    else:
        T, C, H, W = args.seq_length, args.channels, args.height, args.width

    # Create model
    method_class = method_maps[args.method]

    # Build method with minimal config
    model = method_class(
        steps_per_epoch=100,
        test_mean=cfg.get('test_mean', 0.0),
        test_std=cfg.get('test_std', 1.0),
        save_dir=args.save_dir,
        **cfg.__dict__
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.model.load_state_dict(new_state_dict, strict=False)
    else:
        model.model.load_state_dict(checkpoint)

    model.model.eval()
    logger.info(f"Model loaded: {args.method}")

    return model.model, cfg


def create_dummy_input(args):
    """Create dummy input tensor for export."""
    # Input shape: (B, T, C, H, W)
    dummy_input = torch.randn(
        args.batch_size,
        args.seq_length,
        args.channels,
        args.height,
        args.width,
        dtype=torch.float32
    )
    return dummy_input


def export_to_tensorrt(model, dummy_input, args):
    """Export PyTorch model to TensorRT using torch2trt."""
    if not TORCH2TRT_AVAILABLE:
        logger.error("torch2trt is not available. Please install it.")
        return None

    logger.info(f"Exporting model to TensorRT with precision={args.precision}")
    logger.info(f"Input shape: {dummy_input.shape}")

    # Move to CUDA
    model = model.cuda()
    dummy_input = dummy_input.cuda()

    # Set precision flags
    fp16_mode = args.precision == 'fp16'
    int8_mode = args.precision == 'int8'

    if fp16_mode:
        # Check GPU support
        capability = torch.cuda.get_device_capability()
        if capability[0] < 6:
            logger.warning("FP16 requires compute capability >= 6.0")
            fp16_mode = False

    # Export with torch2trt
    logger.info("Running torch2trt conversion...")
    model_trt = torch2trt(
        model,
        [dummy_input],
        fp16_mode=fp16_mode,
        int8_mode=int8_mode,
        max_batch_size=args.max_batch_size,
        max_workspace_size=args.max_workspace_size,
        strict_type_constraints=args.strict_type_constraints,
    )

    logger.info("TensorRT export completed!")

    # Log size comparison
    pytorch_size = sum(p.numel() * p.element_size() for p in model.parameters())
    trt_size = len(model_trt.engine.serialize())

    logger.info(f"PyTorch model size: {pytorch_size / 1024 / 1024:.2f} MB")
    logger.info(f"TensorRT engine size: {trt_size / 1024 / 1024:.2f} MB")

    return model_trt


def save_tensorrt_engine(model_trt, args):
    """Save TensorRT engine to disk."""
    os.makedirs(args.save_dir, exist_ok=True)

    # Save torch2trt model
    trt_path = osp.join(args.save_dir, f'model_trt_{args.precision}.pth')
    torch.save(model_trt.state_dict(), trt_path)
    logger.info(f"Saved TensorRT model to: {trt_path}")

    # Save config for inference
    config_path = osp.join(args.save_dir, 'export_config.json')
    config = {
        'method': args.method,
        'precision': args.precision,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'channels': args.channels,
        'height': args.height,
        'width': args.width,
        'max_batch_size': args.max_batch_size,
    }
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved export config to: {config_path}")

    return trt_path


def validate_export(model_pytorch, model_trt, dummy_input, args):
    """Validate TensorRT model against PyTorch."""
    logger.info("Validating TensorRT export...")

    model_pytorch = model_pytorch.cuda().eval()
    dummy_input = dummy_input.cuda()

    with torch.no_grad():
        output_pytorch = model_pytorch(dummy_input)
        output_trt = model_trt(dummy_input)

    # Calculate difference
    diff = torch.abs(output_pytorch - output_trt)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")

    if max_diff < args.tolerance:
        logger.info("✓ Validation PASSED!")
        return True
    else:
        logger.warning(f"✗ Validation FAILED (tolerance: {args.tolerance})")
        return False


def benchmark_inference(model_pytorch, model_trt, dummy_input, args, num_runs=100):
    """Benchmark inference speed comparison."""
    logger.info(f"Benchmarking inference speed ({num_runs} runs)...")

    model_pytorch = model_pytorch.cuda().eval()
    dummy_input = dummy_input.cuda()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_pytorch(dummy_input)
        _ = model_trt(dummy_input)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_pytorch(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs * 1000

    # Benchmark TensorRT
    start = time.time()
    for _ in range(num_runs):
        _ = model_trt(dummy_input)
    torch.cuda.synchronize()
    trt_time = (time.time() - start) / num_runs * 1000

    logger.info(f"PyTorch inference time: {pytorch_time:.2f} ms")
    logger.info(f"TensorRT inference time: {trt_time:.2f} ms")
    logger.info(f"Speedup: {pytorch_time / trt_time:.2f}x")


def main():
    args = parse_args()

    logger.info("="*60)
    logger.info("OpenSTL TensorRT Export Tool")
    logger.info("="*60)

    # Check requirements
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. TensorRT export requires GPU.")
        return

    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    capability = torch.cuda.get_device_capability()
    logger.info(f"Compute Capability: {capability[0]}.{capability[1]}")

    # Load model
    model, cfg = load_model_and_checkpoint(args)

    # Create dummy input
    dummy_input = create_dummy_input(args)
    logger.info(f"Dummy input shape: {dummy_input.shape}")

    # Export to TensorRT
    model_trt = export_to_tensorrt(model, dummy_input, args)

    if model_trt is None:
        logger.error("Export failed!")
        return

    # Save model
    save_tensorrt_engine(model_trt, args)

    # Validate
    if args.validate:
        validate_export(model, model_trt, dummy_input, args)

    # Benchmark
    benchmark_inference(model, model_trt, dummy_input, args)

    logger.info("="*60)
    logger.info("Export completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
