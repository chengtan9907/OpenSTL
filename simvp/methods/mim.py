import torch.nn as nn

from simvp.models import MIM_Model
from .predrnn import PredRNN


class MIM(PredRNN):
    r"""MIM

    Implementation of `Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    <https://arxiv.org/abs/1811.07490>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device, steps_per_epoch)
        assert args.batch_size == args.val_batch_size, f"{args.batch_size} != {args.val_batch_size}"
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MIM_Model(num_layers, num_hidden, args).to(self.device)
