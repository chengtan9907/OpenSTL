"""Test metrics and evaluation."""
import pytest
import numpy as np

from openstl.core.metrics import metric, MSE, MAE


class TestMetrics:
    """Test metric functions."""

    def test_mse_basic(self):
        """Test MSE calculation."""
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        true = np.array([[1.0, 2.0], [3.0, 4.0]])

        mse = MSE(pred, true)
        assert mse == 0.0, "MSE should be 0 for identical tensors"

    def test_mae_basic(self):
        """Test MAE calculation."""
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        true = np.array([[1.0, 2.0], [3.0, 4.0]])

        mae = MAE(pred, true)
        assert mae == 0.0, "MAE should be 0 for identical tensors"

    def test_mse_nonzero(self):
        """Test MSE with non-zero error."""
        pred = np.array([[0.0, 0.0]])
        true = np.array([[1.0, 1.0]])

        mse = MSE(pred, true)
        assert mse == 1.0, f"MSE should be 1.0, got {mse}"

    def test_mae_nonzero(self):
        """Test MAE with non-zero error."""
        pred = np.array([[0.0, 0.0]])
        true = np.array([[1.0, 1.0]])

        mae = MAE(pred, true)
        assert mae == 1.0, f"MAE should be 1.0, got {mae}"

    def test_metric_function(self):
        """Test the main metric function."""
        pred = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)
        true = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)

        eval_res, eval_log = metric(pred, true, mean=0.0, std=1.0)

        assert 'mae' in eval_res
        assert 'mse' in eval_res
        assert eval_res['mae'] > 0
        assert eval_res['mse'] > 0

    def test_metric_with_normalization(self):
        """Test metric with denormalization."""
        pred = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)
        true = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)

        eval_res, eval_log = metric(
            pred, true,
            mean=0.5,
            std=0.2,
            spatial_norm=True
        )

        assert 'mae' in eval_res
        assert 'mse' in eval_res

    def test_spatial_normalization(self):
        """Test metric with spatial normalization."""
        pred = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)
        true = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)

        eval_res, _ = metric(pred, true, spatial_norm=True)
        eval_res_no_norm, _ = metric(pred, true, spatial_norm=False)

        # Spatial norm should give smaller values (normalized by H*W)
        assert eval_res['mae'] < eval_res_no_norm['mae']

    def test_multiple_metrics(self):
        """Test multiple metrics at once."""
        pred = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)
        true = np.random.randn(4, 10, 1, 32, 32).astype(np.float32)

        eval_res, _ = metric(pred, true, metrics=['mae', 'mse', 'psnr'])

        assert 'mae' in eval_res
        assert 'mse' in eval_res
        assert 'psnr' in eval_res
