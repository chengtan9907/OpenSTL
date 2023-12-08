from .predrnn import PredRNN
from openstl.models import E3DLSTM_Model


class E3DLSTM(PredRNN):
    r"""E3D-LSTM

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, **args):
        PredRNN.__init__(self, **args)

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return E3DLSTM_Model(num_layers, num_hidden, self.hparams)