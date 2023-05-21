import time
import torch
from tqdm import tqdm
import numpy as np
from timm.utils import AverageMeter

from openstl.core import metric
from openstl.models import DMVFN_Model
from openstl.utils import reduce_tensor, LapLoss, VGGPerceptualLoss, ProgressBar, gather_tensors_batch
from openstl.methods.base_method import Base_method


class DMVFN(Base_method):
    r"""DMVFN

    Implementation of `DMVFN: A Dynamic Multi-Scale Voxel Flow Network for Video Prediction
    Predictive Learning <https://arxiv.org/abs/2303.09875>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        max_levels = 5 if args.in_shape[-1] > 32 else 3
        self.lap = LapLoss(max_levels=max_levels, channels=args.in_shape[1])
        self.vggloss = VGGPerceptualLoss(device)

    def _build_model(self, args):
        in_planes = args.in_planes
        num_features = args.num_features
        return DMVFN_Model(in_planes, num_features, args).to(self.device)

    def _predict(self, batch_x, batch_y):
        """Forward the model"""
        merged = self.model(torch.cat([batch_x, batch_y],
                    dim=1), training=self.args.training)
        pred_y = merged[-1]
        batch_y = batch_y.squeeze(1) # (B, C, H, W)
        loss_l1, loss_vgg = 0, 0
        for i in range(self.args.num_block):
            loss_l1 += (self.lap(merged[i], batch_y)).mean() \
                       * (self.args.gamma ** (self.args.num_block - i - 1))
        if self.args.in_shape[1] in [1, 3]:
            loss_vgg = (self.vggloss(pred_y, batch_y)).mean()
        else:
            loss_vgg = (self.vggloss(pred_y.mean(dim=1, keepdim=True),
                                     batch_y.mean(dim=1, keepdim=True))).mean()
        loss_G = loss_l1 + loss_vgg * self.args.coef
        return pred_y, loss_G

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
                imgs = torch.cat([batch_x, batch_y], dim=1)
                b, n, c, h, w = imgs.shape
                loss_avg = 0
                for i in range(n - 2): # Calculate gradients for each time step
                    img0, img1, gt = imgs[:, i].unsqueeze(1), imgs[:, i + 1].unsqueeze(1), imgs[:, i + 2].unsqueeze(1)
                    x = torch.cat([img0, img1], dim=1)
                    pred_y, loss_G = self._predict(x, gt)
                    loss_avg += loss_G

                    if not self.dist:
                        losses_m.update(loss_G.item(), batch_x.size(0))

                    if self.loss_scaler is not None:
                        if torch.any(torch.isnan(loss_G)) or torch.any(torch.isinf(loss_G)):
                            raise ValueError("Inf or nan loss value. Please use fp32 training!")
                        self.loss_scaler(
                            loss_G, self.model_optim,
                            clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                            parameters=self.model.parameters())
                    else:
                        loss_G.backward()
                        self.clip_grads(self.model.parameters())

                    self.model_optim.step()
                    torch.cuda.synchronize()
                    num_updates += 1

                    if self.dist:
                        losses_m.update(reduce_tensor(loss_G), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            loss_avg /= n-2

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss_avg)
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta


    def _inference(self, img0, img1, length=10):
        pred_y = []
        for i in range(length):
            merged = self.model(torch.cat([img0, img1], dim=1), training=False)
            if len(merged) == 0:
                pred = img0
            else:
                pred = merged[-1].unsqueeze(1)
            pred_y.append(pred)

            img0 = img1
            img1 = pred

        pred_y = torch.cat(pred_y, dim=1)
        return pred_y


    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                img0, img1 = batch_x[:, -2:-1], batch_x[:, -1:]
                pred_y = self._inference(img0, img1, length=batch_y.shape[1])

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                B, T, C, H, W = pred_y.shape
                pred_y = pred_y.view(B*T, C, H, W)
                batch_y = batch_y.view(B*T, C, H, W)
                eval_res['loss'] = self.lap(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            # gether tensors by GPU (it's no need to empty cache)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        return results_all

    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):

            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                img0, img1 = batch_x[:, -2:-1], batch_x[:, -1:]
                pred_y = self._inference(img0, img1, length=batch_y.shape[1])

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                B, T, C, H, W = pred_y.shape
                pred_y = pred_y.view(B*T, C, H, W)
                batch_y = batch_y.view(B*T, C, H, W)
                eval_res['loss'] = self.lap(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        return results_all
