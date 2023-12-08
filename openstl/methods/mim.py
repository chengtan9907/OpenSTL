from .predrnn import PredRNN
from openstl.models import MIM_Model


class MIM(PredRNN):
    r"""MIM

    Implementation of `Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    <https://arxiv.org/abs/1811.07490>`_.

    """

    def __init__(self, **args):
        PredRNN.__init__(self, **args)
        assert self.hparams.batch_size == self.hparams.val_batch_size, f"{self.hparams.batch_size} != {self.hparams.val_batch_size}"

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MIM_Model(num_layers, num_hidden, self.hparams)