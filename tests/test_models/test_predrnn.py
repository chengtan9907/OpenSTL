import pytest

from openstl.models import PredRNN_Model


def test_assertion():
    with pytest.raises(AssertionError):
        PredRNN_Model(arch='unknown')

    with pytest.raises(AssertionError):
        # HorNet arch dict should include 'embed_dims',
        PredRNN_Model(arch=dict(base_dim=64))

    with pytest.raises(AssertionError):
        # HorNet arch dict should include 'embed_dims',
        PredRNN_Model(
            arch=dict(
                base_dim=64,
                depths=[2, 3, 18, 2],
                orders=[2, 3, 4, 5],
                dw_cfg=[dict(type='DW', kernel_size=7)] * 4,
            )
        )


def test_convlstm():

    # Test forward
    model = PredRNN_Model(num_layers=3, num_hidden=1)
    model.init_weights()
    model.train()
