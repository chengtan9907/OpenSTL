import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.models import CrevNet_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method


class CrevNet(Base_method):
    r"""CrevNet

    Implementation of `Efficient and Information-Preserving Future Frame Prediction
    and Beyond <https://openreview.net/forum?id=B1eY_pVYvB>`_.
    """

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
        self.model_optim, self.scheduler, self.by_epoch_1 = get_optim_scheduler(
            self.args, self.args.epoch, self.model.frame_predictor, steps_per_epoch)
        self.model_optim2, self.scheduler2, self.by_epoch_2 = get_optim_scheduler(
            self.args, self.args.epoch, self.model.encoder, steps_per_epoch)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        input = torch.cat([batch_x, batch_y], dim=1)
        pred_y, _ = self.model(input, training=False, return_loss=False)
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch_1:
            self.scheduler.step(epoch)
        if self.by_epoch_2:
            self.scheduler2.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()
            self.model_optim2.zero_grad()

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            input = torch.cat([batch_x, batch_y], dim=1)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                loss = self.model(input, training=True)

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
            self.model_optim2.step()
            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch_1:
                self.scheduler.step()
            if not self.by_epoch_2:
                self.scheduler2.step()
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
