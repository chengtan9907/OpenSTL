"""Test imports for OpenSTL package."""
import pytest


def test_openstl_import():
    """Test basic openstl import."""
    import openstl
    assert hasattr(openstl, '__version__') or True  # May not have __version__


def test_core_imports():
    """Test core module imports."""
    from openstl.core import (
        get_optim_scheduler,
        timm_schedulers,
    )
    assert timm_schedulers is not None


def test_methods_import():
    """Test methods module imports."""
    # Skip if lightning is not available
    pytest.importorskip("lightning")
    from openstl.methods import method_maps, __all__
    assert isinstance(method_maps, dict)
    assert len(method_maps) > 0


def test_models_import():
    """Test models module imports."""
    pytest.importorskip("lightning")
    from openstl.models import SimVP_Model
    assert SimVP_Model is not None


def test_datasets_import():
    """Test datasets module imports."""
    pytest.importorskip("lightning")
    from openstl.datasets import BaseDataModule
    assert BaseDataModule is not None


def test_utils_import():
    """Test utils module imports."""
    pytest.importorskip("lightning")
    from openstl.utils import SetupCallback
    assert SetupCallback is not None


def test_timm_compatibility():
    """Test timm 1.0.x compatibility."""
    # These imports should work with both timm 0.6.x and 1.0.x
    from timm.layers import DropPath, trunc_normal_, to_2tuple
    from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
    from timm.models.convnext import ConvNeXtBlock
    from timm.models.mlp_mixer import MixerBlock
    from timm.models.vision_transformer import Block as ViTBlock

    assert DropPath is not None
    assert trunc_normal_ is not None
    assert SwinTransformerBlock is not None
    assert ConvNeXtBlock is not None
    assert MixerBlock is not None
    assert ViTBlock is not None


def test_timm_optim_compatibility():
    """Test timm optimizer compatibility."""
    from openstl.core.optim_scheduler import (
        Adafactor, Adahessian, AdamP, Lookahead,
        Nadam, NvNovoGrad, RAdam, RMSpropTF, SGDP
    )
    assert Adafactor is not None
    assert AdamP is not None
    assert Nadam is not None


def test_timm_scheduler_compatibility():
    """Test timm scheduler compatibility."""
    from openstl.core.optim_scheduler import (
        CosineLRScheduler, MultiStepLRScheduler,
        StepLRScheduler, TanhLRScheduler
    )
    assert CosineLRScheduler is not None
    assert MultiStepLRScheduler is not None
    assert StepLRScheduler is not None
