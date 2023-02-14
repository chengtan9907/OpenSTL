from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from models import CrevNet_Model
from .base_method import Base_method
from .optim_scheduler import get_optim_scheduler
from timm.utils import AverageMeter


class CrevNet(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        args.pre_seq_length = 8
        args.total_length = args.pre_seq_length + args.aft_seq_length
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return CrevNet_Model(**config).to(self.device)

    def _init_optimizer(self, steps_per_epoch):
        self.model_optim, self.scheduler = get_optim_scheduler(self.args, self.args.epoch, self.model.frame_predictor, steps_per_epoch)
        self.model_optim2, self.scheduler2 = get_optim_scheduler(self.args, self.args.epoch, self.model.encoder, steps_per_epoch)

    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()

        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            self.model_optim.zero_grad()
            self.model_optim2.zero_grad()

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            input = torch.cat([batch_x, batch_y], dim=1)
            loss = self.model(input, training=True)
            loss.backward()

            self.model_optim.step()
            self.model_optim2.step()
            
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))
            self.scheduler.step()
            self.scheduler2.step()
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item() / (self.args.pre_seq_length + self.args.aft_seq_length)))

        return num_updates, loss_mean

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            input = torch.cat([batch_x, batch_y], dim=1)
            pred_y, loss = self.model(input, training=False)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [pred_y, batch_y], [preds_lst, trues_lst]))

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
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            input = torch.cat([batch_x, batch_y], dim=1)
            pred_y, _ = self.model(input, training=False)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds