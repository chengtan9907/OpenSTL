import torch
from .base_method import Base_method
from openstl.models import MAU_Model
from openstl.utils import schedule_sampling


class MAU(Base_method):
    r"""MAU

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.eta = 1.0

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MAU_Model(num_layers, num_hidden, self.hparams)
    
    def forward(self, batch_x, batch_y, **kwargs):
        _, img_channel, img_height, img_width = self.hparams.in_shape
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.hparams.total_length - self.hparams.pre_seq_length - 1,
            img_height // self.hparams.patch_size,
            img_width // self.hparams.patch_size,
            self.hparams.patch_size ** 2 * img_channel)).to(self.device)
        img_gen, _ = self.model(test_ims, real_input_flag, return_loss=False)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:, :]
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        eta, real_input_flag = schedule_sampling(self.eta, self.global_step, ims.shape[0], self.hparams)
        img_gen, loss = self.model(ims, real_input_flag)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss