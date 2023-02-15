from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from models import SimVP_Model
from .base_method import Base_method
from timm.utils import AverageMeter


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model(**config).to(self.device)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()

        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self._predict(batch_x)

            loss = self.criterion(pred_y, batch_y)
            loss.backward()
            self.model_optim.step()
            self.scheduler.step()
            
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))

            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, loss_mean

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self._predict(batch_x)
            loss = self.criterion(pred_y, batch_y)

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
        for batch_x, batch_y in test_pbar:
            pred_y = self._predict(batch_x.to(self.device))

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds
