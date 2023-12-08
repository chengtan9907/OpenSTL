from .predrnn import PredRNN
from openstl.models import ConvLSTM_Model


class ConvLSTM(PredRNN):
    r"""ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, **args):
        PredRNN.__init__(self, **args)

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return ConvLSTM_Model(num_layers, num_hidden, self.hparams)