from typing import Dict, List, Union

import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import dist_forward_collect, nondist_forward_collect

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank = int(device.split(':')[-1])
        else:
            self.rank = 0
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.args.rank == 0:
               print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank], broadcast_buffers=True)

    def train_one_epoch(self, runner, train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def forward_test(self, batch_x, batch_y):
        """Evaluate the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.

        Returns:
            dict(tensor): The concatenated outputs with keys.
        """
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        with self.amp_autocast():
            pred_y = self._predict(batch_x)

        return dict(zip(['inputs', 'preds', 'trues'],
                        [batch_x, pred_y, batch_y]))

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
        """
        self.model.eval()
        func = lambda *x: self.forward_test(*x)
        if self.dist:
            results = dist_forward_collect(func, vali_loader, self.rank,
                                           len(vali_loader.dataset), to_numpy=True)
        else:
            results = nondist_forward_collect(func, vali_loader,
                                              len(vali_loader.dataset), to_numpy=True)

        preds = torch.tensor(results['preds']).to(self.device)
        trues = torch.tensor(results['trues']).to(self.device)
        losses_m = self.criterion(preds, trues).cpu().numpy()
        return results['preds'], results['trues'], losses_m

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        func = lambda *x: self.forward_test(*x)
        if self.dist:
            results = dist_forward_collect(func, test_loader, self.rank,
                                           len(test_loader.dataset), to_numpy=True)
        else:
            results = nondist_forward_collect(func, test_loader,
                                              len(test_loader.dataset), to_numpy=True)

        return results['inputs'], results['preds'], results['trues']

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."
