# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### P2 - TensorRT Deployment (Completed)
- Added `tools/export_to_trt.py` - Model export script for TensorRT
- Added `tools/inference_trt.py` - TensorRT inference script
- Added `docs/TENSORRT_DEPLOYMENT.md` - Comprehensive deployment guide
- Support for FP32, FP16, INT8 precision modes
- Built-in validation and benchmarking tools

### P0 - Code Optimization & Compatibility (Completed)
- Added `pyproject.toml` for modern Python packaging
- Added GitHub Actions CI/CD workflow (`.github/workflows/tests.yml`)
- Created comprehensive pytest test infrastructure:
  - `tests/test_imports.py` - Import compatibility tests
  - `tests/test_methods/test_registration.py` - Method registration tests
  - `tests/test_models/test_instantiation.py` - Model instantiation tests
  - `tests/test_datasets/test_dataloaders.py` - Data loader tests
  - `tests/test_core/test_training.py` - Training loop tests
  - `tests/test_core/test_metrics.py` - Metric function tests
  - `tests/test_core/test_optim_scheduler.py` - Optimizer/scheduler tests
  - `tests/conftest.py` - Pytest fixtures and configuration

### P1 - Benchmark Unification (Completed)
- Added configuration templates under `configs/templates/`:
  - `base_config.py` - Base template with full documentation
  - `simvp_mmnist.py` - SimVP Moving MNIST reference
  - `predrnn_mmnist.py` - PredRNN Moving MNIST reference
- Added structured output (JSON/CSV) in `openstl/methods/base_method.py`
- Added training history CSV logging in `openstl/utils/callbacks.py`

### Changed
- Updated `requirements/runtime.txt` with flexible version ranges:
  - `lightning>=2.2.1,<3.0`
  - `timm>=0.9.0,<2.0`
- Updated `environment.yml` with modern dependency versions:
  - `python>=3.8,<3.11`
  - `pytorch>=2.1.0`

### Added
- AMP/FP8 precision support in `openstl/api/exp.py`:
  - `precision` parameter ('16-mixed', 'bf16-mixed', 'fp8')
  - Gradient clipping support via `grad_clip` and `grad_clip_algorithm`

### Fixed
- **timm 1.0.x compatibility** in `openstl/core/optim_scheduler.py`:
  - Added try-except blocks for optimizer imports
  - Handled optimizer name changes (e.g., `Nadam` → `NAdam`)
- **timm 1.0.x compatibility** in `openstl/modules/simvp_modules.py`:
  - Added fallback import for `ConvNeXtBlock`
- **timm 1.0.x compatibility** in `openstl/modules/wast_modules.py`:
  - Added fallback import for `SqueezeExcite`, `InvertedResidual`

## [1.0.0] - 2023-XX-XX

### Added
- Initial release of OpenSTL
