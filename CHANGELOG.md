# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pyproject.toml` for modern Python packaging
- GitHub Actions CI/CD workflow (`.github/workflows/tests.yml`)
- Comprehensive pytest test infrastructure:
  - `tests/test_imports.py` - Import compatibility tests
  - `tests/test_methods/test_registration.py` - Method registration tests
  - `tests/test_models/test_instantiation.py` - Model instantiation tests
  - `tests/test_datasets/test_dataloaders.py` - Data loader tests
  - `tests/conftest.py` - Pytest fixtures and configuration

### Changed
- Updated `requirements/runtime.txt` with flexible version ranges:
  - `lightning>=2.2.1,<3.0`
  - `timm>=0.9.0,<2.0`
- Updated `environment.yml` with modern dependency versions:
  - `python>=3.8,<3.11`
  - `pytorch>=2.1.0`
  - Flexible `lightning` and `timm` version ranges

### Fixed
- **timm 1.0.x compatibility** in `openstl/core/optim_scheduler.py`:
  - Added try-except blocks for optimizer imports to handle API changes
  - Support for both old (`timm.optim.xxx`) and new (`timm.optim.xxx`) import paths
  - Handled optimizer name changes (e.g., `Nadam` → `NAdam` in newer timm)
- **timm 1.0.x compatibility** in `openstl/modules/simvp_modules.py`:
  - Added fallback import for `ConvNeXtBlock`
- **timm 1.0.x compatibility** in `openstl/modules/wast_modules.py`:
  - Added fallback import for `SqueezeExcite` and `InvertedResidual`

## [1.0.0] - 2023-XX-XX

### Added
- Initial release of OpenSTL
