import torch
import numpy as np

from .base_method import Base_method
from openstl.models import PhyDNet_Model


class PhyDNet(Base_method):
    r"""PhyDNet

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.constraints = self._get_constraints()

    def _build_model(self, **args):
        return PhyDNet_Model(self.hparams)

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def forward(self, batch_x, batch_y, **kwargs):
        if not self.hparams.dist:
            pred_y, _ = self.model.inference(batch_x, batch_y, self.constraints, return_loss=False)
        else:
            pred_y, _ = self.model.module.inference(batch_x, batch_y, self.constraints, return_loss=False)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        teacher_forcing_ratio = np.maximum(0 , 1 - self.current_epoch * 0.003) 
        pred_y = self.model(batch_x, batch_y, self.constraints, teacher_forcing_ratio)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss