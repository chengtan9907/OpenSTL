import torch.nn as nn

from models import MIM_Model
from .predrnn import PredRNN

from utils import *


class MIM(PredRNN):
    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device, steps_per_epoch)
        assert args.batch_size == args.val_batch_size
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MIM_Model(num_layers, num_hidden, args).to(self.device)