from .predrnn import PredRNN
from openstl.models import PredRNNv2_Model


class PredRNNv2(PredRNN):
    r"""PredRNNv2

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, **args):
        PredRNN.__init__(self, **args)

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNv2_Model(num_layers, num_hidden, self.hparams)