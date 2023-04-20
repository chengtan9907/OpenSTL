import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import MAU_Model
from openstl.utils import reduce_tensor, schedule_sampling
from .base_method import Base_method


class MAU(Base_method):
    r"""MAU

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MAU_Model(num_layers, num_hidden, args).to(self.device)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model."""
        _, img_channel, img_height, img_width = self.args.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.args.total_length - self.args.pre_seq_length - 1,
            img_height // self.args.patch_size,
            img_width // self.args.patch_size,
            self.args.patch_size ** 2 * img_channel)).to(self.device)

        img_gen, _ = self.model(test_ims, real_input_flag, return_loss=False)
        pred_y = img_gen[:, -self.args.aft_seq_length:, :]

        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            # preprocess
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            eta, real_input_flag = schedule_sampling(eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)

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
