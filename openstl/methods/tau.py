import torch
import torch.nn.functional as F

from .simvp import SimVP
from openstl.models import SimVP_Model


class TAU(SimVP):
    r"""TAU

    Implementation of `Temporal Attention Unit: Towards Efficient Spatiotemporal 
    Predictive Learning <https://arxiv.org/abs/2206.12126>`_.

    """

    def __init__(self, **args):
        SimVP.__init__(self, **args)

    def _build_model(self, **args):
        return SimVP_Model(**args).to(self.device)
    
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y) + self.hparams.alpha * self.diff_div_reg(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss