#!/usr/bin/env python
# Copyright (c) CAIRI AI Lab. All rights reserved
"""
TensorRT Inference for OpenSTL models.

This script performs inference using exported TensorRT models.

Usage:
    # Run inference with TensorRT model
    python tools/inference_trt.py \\
        --trt-model work_dirs/trt_export/model_trt_fp16.pth \\
        --config work_dirs/trt_export/export_config.json \\
        --input-path data/sample.npy \\
        --output-path results/prediction.npy
"""

import os
import os.path as osp
import argparse
import logging

import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT Inference for OpenSTL')

    parser.add_argument('--trt-model', type=str, required=True,
                        help='Path to exported TensorRT model (.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to export config file')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to input data (.npy)')
    parser.add_argument('--output-path', type=str, default='prediction.npy',
                        help='Path to save predictions')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')

    args = parser.parse_args()
    return args


def load_tensorrt_model(trt_path):
    """Load TensorRT model."""
    from torch2trt import TRTModule

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_path))

    logger.info(f"Loaded TensorRT model: {trt_path}")
    return model_trt


def load_config(config_path):
    """Load export config."""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config}")
    return config


def main():
    args = parse_args()

    # Load model and config
    model_trt = load_tensorrt_model(args.trt_model)
    config = load_config(args.config)

    # Load input data
    logger.info(f"Loading input data: {args.input_path}")
    input_data = np.load(args.input_path)

    # Ensure correct shape
    expected_shape = (
        config['batch_size'],
        config['seq_length'],
        config['channels'],
        config['height'],
        config['width']
    )

    if input_data.shape != expected_shape:
        logger.warning(f"Input shape {input_data.shape} != expected {expected_shape}")
        # Reshape or pad/crop as needed
        if len(input_data.shape) == 4:
            input_data = input_data[np.newaxis, ...]

    # Convert to tensor
    input_tensor = torch.from_numpy(input_data).float().cuda()

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        output = model_trt(input_tensor)

    output_np = output.cpu().numpy()

    # Save output
    np.save(args.output_path, output_np)
    logger.info(f"Saved predictions to: {args.output_path}")
    logger.info(f"Output shape: {output_np.shape}")


if __name__ == '__main__':
    main()
