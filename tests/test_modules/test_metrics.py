from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

# Load the metric module without initializing OpenSTL's training schedulers.
metrics_module = run_path(
    str(Path(__file__).resolve().parents[2] / 'openstl' / 'core' / 'metrics.py')
)
metric = metrics_module['metric']
sevir_metrics = metrics_module['sevir_metrics']


def assert_weather_metrics(pred, true, threshold):
    valid = ~(np.isnan(pred) | np.isnan(true))
    observed = true >= threshold
    forecast = pred >= threshold
    axes = (0, 2, 3, 4)
    hits = np.count_nonzero(valid & observed & forecast, axis=axes)
    false_alarms = np.count_nonzero(valid & ~observed & forecast, axis=axes)
    misses = np.count_nonzero(valid & observed & ~forecast, axis=axes)

    np.testing.assert_array_equal(
        sevir_metrics(pred, true, threshold), (hits, false_alarms, misses)
    )

    # Preserve the existing epsilon and mean over forecast lead times.
    eps = 1e-6
    expected = {
        'pod': np.mean((hits + eps) / (hits + misses + eps)),
        'sucr': np.mean((hits + eps) / (hits + false_alarms + eps)),
        'csi': np.mean((hits + eps) / (hits + false_alarms + misses + eps)),
    }
    actual, _ = metric(pred, true, metrics=['pod', 'sucr', 'csi'], threshold=threshold)
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize(
    ('prediction', 'observation', 'threshold'),
    [
        pytest.param([100, 100, 0, 100, 0], [100, 0, 100, 0, 0], 74,
                     id='one-hit-two-false-alarms-one-miss'),
        pytest.param([100, 0, 0], [100, 0, 0], 74, id='perfect-sparse'),
        pytest.param([0, 0], [0, 0], 74, id='no-events'),
        pytest.param([100, 100], [100, 100], 74, id='all-events'),
        pytest.param([2.5, 2.49], [2.5, 2.49], 2.5, id='inclusive-boundary'),
        pytest.param([-2, -4, -4], [-1, -2, -5], -2.5, id='negative-threshold'),
        pytest.param([np.nan, 90, 83, 0], [90, np.nan, 82, -1], 74, id='paired-nan-mask'),
    ],
)
def test_sevir_contingency_counts_and_scores(prediction, observation, threshold):
    pred = np.array(prediction, dtype=np.float32).reshape(1, 1, 1, 1, -1)
    true = np.array(observation, dtype=np.float32).reshape(1, 1, 1, 1, -1)
    assert_weather_metrics(pred, true, threshold)


def test_sevir_metrics_keep_forecast_lead_times_separate():
    pred = np.array([[100, 0], [100, 100], [0, 100], [100, 0]], dtype=np.float32)
    true = np.array([[100, 100], [0, 100], [0, 0], [100, 100]], dtype=np.float32)
    assert_weather_metrics(pred.reshape(2, 2, 1, 1, 2), true.reshape(2, 2, 1, 1, 2), 74)


@pytest.mark.parametrize(
    ('spatial_norm', 'mae', 'mse'), [(False, 4.0, 10.0), (True, 2.0, 5.0)]
)
def test_non_weather_error_metrics(spatial_norm, mae, mse):
    pred = np.array([1, -3], dtype=np.float32).reshape(1, 1, 1, 1, 2)
    true = np.zeros_like(pred)
    actual, log = metric(pred, true, metrics=['mae', 'mse', 'rmse'],
                         spatial_norm=spatial_norm, return_log=False)
    assert actual == pytest.approx({'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)})
    assert log == ''
