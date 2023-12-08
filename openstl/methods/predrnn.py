import torch
import torch.nn as nn
from openstl.models import PredRNN_Model
from openstl.utils import (reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method


class PredRNN(Base_method):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.eta = 1.0

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNN_Model(num_layers, num_hidden, self.hparams)

    def forward(self, batch_x, batch_y, **kwargs):
        # reverse schedule sampling
        if self.hparams.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.hparams.pre_seq_length
        _, img_channel, img_height, img_width = self.hparams.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = reshape_patch(test_ims, self.hparams.patch_size)
        test_ims = test_ims[:, :, :, :, :img_channel]

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.hparams.total_length - mask_input - 1,
            img_height // self.hparams.patch_size,
            img_width // self.hparams.patch_size,
            self.hparams.patch_size ** 2 * img_channel)).to(self.device)
            
        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.hparams.pre_seq_length - 1, :, :] = 1.0

        img_gen, _ = self.model(test_dat, real_input_flag, return_loss=False)
        img_gen = reshape_patch_back(img_gen, self.hparams.patch_size)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        ims = reshape_patch(ims, self.hparams.patch_size)

        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(
                self.global_step, ims.shape[0], self.hparams)
        else:
            self.eta, real_input_flag = schedule_sampling(
                self.eta, self.global_step, ims.shape[0], self.hparams)
            
        img_gen, loss = self.model(ims, real_input_flag)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss