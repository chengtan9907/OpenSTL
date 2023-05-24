import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from timm.utils import AverageMeter
from openstl.models import PredNet_Model
from openstl.utils import (reduce_tensor, get_initial_states)
from openstl.methods.base_method import Base_method


class PredNet(Base_method):
    r"""PredNet

    Implementation of `Deep Predictive Coding Networks for Video Prediction
    and Unsupervised Learning <https://arxiv.org/abs/1605.08104>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()
        self.train_loss = TrainLossCalculator(num_layer=len(self.args.stack_sizes), timestep=self.args.pre_seq_length +
                                              self.args.aft_seq_length, weight_mode=self.args.weight_mode, device=self.device)

    def _build_model(self, args):
        return PredNet_Model(args.stack_sizes, args.R_stack_sizes,
                             args.A_filt_sizes, args.Ahat_filt_sizes,
                             args.R_filt_sizes, args.pixel_max, args)

    def _predict(self, batch_x, batch_y, **kwargs):
        input = torch.cat([batch_x, batch_y], dim=1)
        states = get_initial_states(input.shape, -2, -1, len(self.args.stack_sizes),
                                    self.args.R_stack_sizes, self.args.stack_sizes,
                                    -3, self.args.device)
        predict_list, _ = self.model(input, states, extrapolation=True)
        pred_y = torch.stack(predict_list[batch_x.shape[1]:], dim=1)
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

            with self.amp_autocast():
                input = torch.cat([batch_x, batch_y], dim=1)
                states = get_initial_states(input.shape, -2, -1, len(self.args.stack_sizes),
                                            self.args.R_stack_sizes, self.args.stack_sizes,
                                            -3, self.args.device)

                _, error_list = self.model(input, states, extrapolation=False)
                loss = self.train_loss.calculate_loss(error_list)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError(
                        "Inf or nan loss value. Please use fp32 training!")
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


class TrainLossCalculator:
    def __init__(self, num_layer, timestep, weight_mode, device):
        self.num_layers = num_layer
        self.timestep = timestep
        self.weight_mode = weight_mode
        self.device = device

        if self.weight_mode == 'L_0':
            layer_weights = np.array([0. for _ in range(num_layer)])
            layer_weights[0] = 1.
        elif self.weight_mode == 'L_all':
            layer_weights = np.array([0.1 for _ in range(num_layer)])
            layer_weights[0] = 1.
        else:
            raise (RuntimeError('Unknown loss weighting mode! '
                                'Please use `L_0` or `L_all`.'))
        self.layer_weights = torch.from_numpy(layer_weights).to(self.device)

    def calculate_loss(self, input):
        # Weighted by layer
        error_list = [batch_numLayer_error * self.layer_weights for
                      batch_numLayer_error in input]  # Use the broadcast
        error_list = [torch.sum(error_at_t) for error_at_t in error_list]

        # Weighted by timestep
        time_loss_weights = torch.cat([torch.tensor([0.], device=self.device),
                                       torch.full((self.timestep - 1,),
                                                  1. / (self.timestep - 1), device=self.device)])

        total_error = error_list[0] * time_loss_weights[0]
        for err, time_weight in zip(error_list[1:], time_loss_weights[1:]):
            total_error += err * time_weight
        total_error /= input[0].shape[0]  # input[0].shape[0] = B
        return total_error
