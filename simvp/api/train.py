# Copyright (c) CAIRI AI Lab. All rights reserved

import time
import logging
import json
import torch
import numpy as np
import os.path as osp
from typing import Dict, List
from fvcore.nn import FlopCountAnalysis, flop_count_table

from simvp.core import Hook, metric, Recorder, get_priority, hook_maps
from simvp.methods import method_maps
from simvp.utils import (set_seed, print_log, output_namespace, check_dir,
                         get_dataset, get_dist_info, measure_throughput, weights_to_cpu)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


class NonDistExperiment(object):
    """ Experiment with non-dist PyTorch training and evaluation """

    def __init__(self, args):
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self.args.method = self.args.method.lower()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = self.config['epoch']
        self._max_iters = None
        self._hooks: List[Hook] = []

        self._preparation()
        print_log(output_namespace(self.args))

        T, C, H, W = self.args.in_shape
        if self.args.method == 'simvp':
            input_dummy = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
        elif self.args.method == 'crevnet':
            # crevnet must use the batchsize rather than 1
            input_dummy = torch.ones(self.args.batch_size, 20, C, H, W).to(self.device)
        elif self.args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            _tmp_input2 = torch.ones(1, self.args.aft_seq_length, C, H, W).to(self.device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(self.device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        elif self.args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau']:
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.aft_seq_length - 1, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif self.args.method == 'predrnnv2':
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.total_length - 2, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        print_log(self.method.model)
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        print_log(flop_count_table(flops))
        if args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            print_log('Throughputs of {}: {:.3f}'.format(self.args.method, fps))

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.dist:
                self._rank, self._world_size = get_dist_info()
                self.device = f'cuda:{self._rank}'
                print(f'Use GPU: local rank={self._rank}')
            else:
                device = torch.device('cuda:0')
                print('Use GPU:', device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        prefix = 'train' if not self.args.test else 'test'
        logging.basicConfig(level=logging.INFO,
                            filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()
        # build hooks
        self._build_hook()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)
        self.call_hook('before_run')

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _build_hook(self):
        for k in self.args.__dict__:
            if k.lower().endswith('hook'):
                hook_cfg = self.args.__dict__[k].copy()
                priority = get_priority(hook_cfg.pop('priority', 'NORMAL'))
                hook = hook_maps[k.lower()](**hook_cfg)
                if hasattr(hook, 'priority'):
                    raise ValueError('"priority" is a reserved attribute for hooks')
                hook.priority = priority  # type: ignore
                # insert the hook to a sorted list
                inserted = False
                for i in range(len(self._hooks) - 1, -1, -1):
                    if priority >= self._hooks[i].priority:  # type: ignore
                        self._hooks.insert(i + 1, hook)
                        inserted = True
                        break
                if not inserted:
                    self._hooks.insert(0, hook)

    def call_hook(self, fn_name: str) -> None:
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def _get_data(self):
        self.train_loader, self.vali_loader, self.test_loader = get_dataset(self.args.dataname, self.config)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader
        self._max_iters = self._max_epochs * len(self.train_loader)

    def _get_hook_info(self):
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def _save(self, name=''):
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self.method.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def train(self):
        recorder = Recorder(verbose=True)
        num_updates = 0
        self.call_hook('before_train_epoch')
        # constants for other methods:
        eta = 1.0  # PredRNN
        for epoch in range(self._epoch, self._max_epochs):
            loss_mean = 0.0

            if self.args.method in ['simvp', 'crevnet', 'phydnet']:
                num_updates, loss_mean = self.method.train_one_epoch(
                    self, self.train_loader, epoch, num_updates, loss_mean)
            elif self.args.method in ['convlstm', 'predrnnpp', 'predrnn', 'predrnnv2', 'mim', 'e3dlstm', 'mau']:
                num_updates, loss_mean, eta = self.method.train_one_epoch(
                    self, self.train_loader, epoch, num_updates, loss_mean, eta)
            else:
                raise ValueError(f'Invalid method name {self.args.method}')

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)

                print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                    epoch + 1, len(self.train_loader), cur_lr, loss_mean, vali_loss))
                recorder(vali_loss, self.method.model, self.path)
                self._save(name='latest')

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def vali(self, vali_loader):
        self.call_hook('before_val_epoch')
        preds, trues, val_loss = self.method.vali_one_epoch(self, self.vali_loader)
        self.call_hook('after_val_epoch')

        if 'weather' in self.args.dataname:
            metric_list, spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            metric_list, spatial_norm = ['mse', 'mae'], False
        eval_res, eval_log = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        print_log('val\t '+eval_log)
        if has_nni:
            nni.report_intermediate_result(eval_res['mse'])

        return val_loss

    def test(self):
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            self.method.model.load_state_dict(torch.load(best_model_path))

        self.call_hook('before_val_epoch')
        inputs, trues, preds = self.method.test_one_epoch(self, self.test_loader)
        self.call_hook('after_val_epoch')

        if 'weather' in self.args.dataname:
            metric_list, spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            metric_list, spatial_norm = ['mse', 'mae', 'ssim', 'psnr'], False
        eval_res, eval_log = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        metrics = np.array([eval_res['mae'], eval_res['mse']])
        print_log(eval_log)

        folder_path = osp.join(self.path, 'saved')
        check_dir(folder_path)

        for np_data in ['metrics', 'inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return eval_res['mse']
