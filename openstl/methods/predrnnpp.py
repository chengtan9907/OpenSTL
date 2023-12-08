from .predrnn import PredRNN
from openstl.models import PredRNNpp_Model


class PredRNNpp(PredRNN):
    r"""PredRNN++

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, **args):
        PredRNN.__init__(self, **args)

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNpp_Model(num_layers, num_hidden, self.hparams)