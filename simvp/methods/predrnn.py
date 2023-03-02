import torch
import torch.nn as nn
import numpy as np
from timm.utils import AverageMeter
from tqdm import tqdm

from simvp.models import PredRNN_Model
from simvp.utils import (reshape_patch, reshape_patch_back,
                         reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method


class PredRNN(Base_method):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNN_Model(num_layers, num_hidden, args).to(self.device)

    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, eta, **kwargs):
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)

        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            img_gen, loss = self.model(ims, real_input_flag)
            loss.backward()
            self.model_optim.step()
            if not self.by_epoch:
                self.scheduler.step(epoch)

            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))

            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, loss_mean, eta

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)

        # reverse schedule sampling
        if self.args.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.args.pre_seq_length

        _, img_channel, img_height, img_width = self.args.in_shape

        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # preprocess
            test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            test_dat = reshape_patch(test_ims, self.args.patch_size)
            test_ims = test_ims[:, :, :, :, :img_channel]

            real_input_flag = torch.zeros(
                (batch_x.shape[0],
                self.args.total_length - mask_input - 1,
                img_height // self.args.patch_size,
                img_width // self.args.patch_size,
                self.args.patch_size ** 2 * img_channel)).to(self.device)

            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag[:, :self.args.pre_seq_length - 1, :, :] = 1.0

            img_gen, loss = self.model(test_dat, real_input_flag)

            img_gen = reshape_patch_back(img_gen, self.args.patch_size)
            pred_y = img_gen[:, -self.args.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
          
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()
                                                  ), [pred_y, batch_y], [preds_lst, trues_lst]))

            if i * batch_x.shape[0] > 1000:
                break
    
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
        
        total_loss = np.average(total_loss)

        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        return preds, trues, total_loss

    def test_one_epoch(self, test_loader, **kwargs):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(test_loader)

        # reverse schedule sampling
        if self.args.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.args.pre_seq_length

        _, img_channel, img_height, img_width = self.args.in_shape

        for batch_x, batch_y in test_pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # preprocess
            test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            test_dat = reshape_patch(test_ims, self.args.patch_size)
            test_ims = test_ims[:, :, :, :, :img_channel]

            real_input_flag = torch.zeros(
                (batch_x.shape[0],
                self.args.total_length - mask_input - 1,
                img_height // self.args.patch_size,
                img_width // self.args.patch_size,
                self.args.patch_size ** 2 * img_channel)).to(self.device)
                
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag[:, :self.args.pre_seq_length - 1, :, :] = 1.0

            img_gen, _ = self.model(test_dat, real_input_flag)
            img_gen = reshape_patch_back(img_gen, self.args.patch_size)
            pred_y = img_gen[:, -self.args.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds
