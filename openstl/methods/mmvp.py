import torch
from openstl.models import MMVP_Model
from .base_method import Base_method 

class MMVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """
    
    def __init__(self, **args):
        super().__init__(**args)
    
    def _build_model(self, **args):
        return MMVP_Model(**args)
    
    def forward(self, batch_x, batch_y=None, **kwargs):
        pred_y = self.model(batch_x)
        return pred_y 

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(torch.cat((batch_x, batch_y), dim=1))
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss