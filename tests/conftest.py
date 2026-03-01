"""Pytest configuration and fixtures."""
import os
import tempfile
import pytest
import torch


@pytest.fixture(scope='session')
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope='session')
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


@pytest.fixture
def sample_input():
    """Create a sample spatio-temporal input tensor."""
    # (B, T, C, H, W) - batch=2, seq=10, channels=1, 64x64
    return torch.randn(2, 10, 1, 64, 64)


@pytest.fixture
def sample_input_3d():
    """Create a sample 3D spatio-temporal input tensor."""
    # (B, T, C, H, W) - batch=2, seq=4, channels=2, 32x32
    return torch.randn(2, 4, 2, 32, 32)


@pytest.fixture
def minimal_config():
    """Create minimal configuration for testing."""
    return {
        'seed': 42,
        'epoch': 1,
        'lr': 0.001,
        'weight_decay': 0.0,
        'batch_size': 2,
    }


@pytest.fixture(autouse=True)
def set_env_vars():
    """Set environment variables for testing."""
    os.environ['OPENSTL_TEST'] = '1'
    yield
    if 'OPENSTL_TEST' in os.environ:
        del os.environ['OPENSTL_TEST']
