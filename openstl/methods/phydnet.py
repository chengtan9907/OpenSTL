import time
import torch
import torch.nn as nn
import numpy as np
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PhyDNet_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method


class PhyDNet(Base_method):
    r"""PhyDNet

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
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

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        if not self.dist:
            pred_y, _ = self.model.inference(batch_x, batch_y, self.constraints, return_loss=False)
        else:
            pred_y, _ = self.model.module.inference(batch_x, batch_y, self.constraints, return_loss=False)
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y = self.model(batch_x, batch_y, self.constraints, teacher_forcing_ratio)
                loss = self.criterion(pred_y, batch_y)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())

            self.model_optim.step()
            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
