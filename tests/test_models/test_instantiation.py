"""Test model instantiation and basic forward pass."""
import pytest

# Skip all tests in this module if lightning is not available
pytest.importorskip("lightning")

import torch
from openstl.models import SimVP_Model


class TestSimVPModel:
    """Test SimVP model instantiation and forward pass."""

    def test_simvp_creation_valid(self):
        """Test SimVP model creation with valid architecture."""
        # in_shape = (T, C, H, W)
        model = SimVP_Model(in_shape=(10, 1, 64, 64), hid_S=16, N_S=4, N_T=4)
        assert model is not None

    def test_simvp_creation_invalid_arch(self):
        """Test SimVP model creation with invalid architecture."""
        with pytest.raises((AssertionError, ValueError)):
            # arch parameter is no longer used, but test with invalid input
            SimVP_Model(in_shape=(10, 1, 64, 64), model_type='unknown_arch_test')

    def test_simvp_forward_pass(self):
        """Test SimVP model forward pass."""
        model = SimVP_Model(in_shape=(10, 1, 64, 64), hid_S=16, N_S=4, N_T=4)
        model.train()

        # Create dummy input: (B, T, C, H, W)
        dummy_input = torch.randn(2, 10, 1, 64, 64)

        # Forward pass
        output = model(dummy_input)

        # Check output shape matches expected
        assert output.shape[0] == 2  # batch size
        assert len(output.shape) == 5  # (B, T, C, H, W)

    def test_simvp_init_weights(self):
        """Test SimVP model weight initialization."""
        model = SimVP_Model(in_shape=(10, 1, 64, 64), hid_S=16, N_S=4, N_T=4)
        # SimVP uses torch's default initialization
        # Just verify model can be created and has parameters
        assert sum(p.numel() for p in model.parameters()) > 0


class TestOtherModels:
    """Test other model classes if available."""

    def test_model_module_structure(self):
        """Test that model module has expected structure."""
        from openstl import models

        # Check that SimVP_Model is exported
        assert hasattr(models, 'SimVP_Model')
