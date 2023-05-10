import time
import torch
import torch.nn as nn
import numpy as np
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PredNet_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method


class PredNet(Base_method):
    r"""PredNet

    Implementation of `Deep Predictive Coding Networks for Video Prediction
    and Unsupervised Learning <https://arxiv.org/abs/1605.08104>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()
        
        self.constraints = self._get_constraints()

    def _build_model(self, args):
        return PredNet_Model(args, output_mode='error').to(self.device)

    def train_one_epoch(self, runner, train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError
