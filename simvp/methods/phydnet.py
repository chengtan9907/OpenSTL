import torch
import torch.nn as nn
import numpy as np
from timm.utils import AverageMeter
from tqdm import tqdm

from simvp.models import PhyDNet_Model
from .base_method import Base_method


class PhyDNet(Base_method):
    r"""PhyDNet

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()
        
        self.constraints = self._get_constraints()

    def _build_model(self, config):
        return PhyDNet_Model(**config).to(self.device)

    def _get_constraints(self):
        constraints = torch.zeros((49,7,7)).to(self.device)
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()

        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 

        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x, batch_y, self.constraints, teacher_forcing_ratio)
            loss = self.criterion(pred_y, batch_y)  
            loss.backward()
            self.model_optim.step()
            
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))
            self.scheduler.step(epoch)
            train_pbar.set_description('train loss: {:.4f}'.format(
                loss.item() / (self.args.pre_seq_length + self.args.aft_seq_length)))

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, loss_mean

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y, loss = self.model.inference(batch_x, batch_y, self.constraints)
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
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y, _ = self.model.inference(batch_x, batch_y, self.constraints)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds
