import torch.nn as nn

from models import E3DLSTM_Model
from .predrnn import PredRNN

from utils import *


class E3DLSTM(PredRNN):
    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return E3DLSTM_Model(num_layers, num_hidden, args).to(self.device)