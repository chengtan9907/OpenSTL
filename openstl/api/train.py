# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import os.path as osp
import time
import logging
import json
import numpy as np
from typing import Dict, List
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
import torch.distributed as dist

from openstl.core import Hook, metric, Recorder, get_priority, hook_maps
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, output_namespace, check_dir, collect_env,
                           init_dist, init_random_seed,
                           get_dataset, get_dist_info, measure_throughput, weights_to_cpu)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args):
        """Initialize experiments (non-dist as an example)"""
        self.args = args
        self.config = self.args.__dict__
        self.device = self.args.device
        self.method = None
        self.args.method = self.args.method.lower()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = self.config['epoch']
        self._max_iters = None
        self._hooks: List[Hook] = []
        self._rank = 0
        self._world_size = 1
        self._dist = self.args.dist

        self._preparation()
        if self._rank == 0:
            print_log(output_namespace(self.args))
            self.display_method_info()

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self.args.dist:
                device = f'cuda:{self._rank}'
                torch.cuda.set_device(self._rank)
                print(f'Use distributed mode with GPUs: local rank={self._rank}')
            else:
                device = torch.device('cuda:0')
                print('Use non-distributed mode with GPU:', device)
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print('Use CPU')
            if self.args.dist:
                assert False, "Distributed training requires GPUs"
        return device

    def _preparation(self):
        """Preparation of environment and basic experiment setups"""
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.args.local_rank)

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir+'/')[-1])
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

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True
        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self._rank, self._world_size = get_dist_info()
            # re-set gpu_ids with distributed training mode
            self._gpu_ids = range(self._world_size)
        self.device = self._acquire_device()

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        if self._rank == 0:
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

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
        self.steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, self.steps_per_epoch)
        self.method.model.eval()
        # setup ddp training
        if self._dist:
            self.method.model.cuda()
            if self.args.torchscript:
                self.method.model = torch.jit.script(self.method.model)
            self.method._init_distributed()

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
        """Run hooks by the registered names"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def _get_hook_info(self):
        """Get hook information in each stage"""
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self._hooks:
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

    def _get_data(self):
        """Prepare datasets and dataloaders"""
        self.train_loader, self.vali_loader, self.test_loader = \
            get_dataset(self.args.dataname, self.config)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader
        self._max_iters = self._max_epochs * len(self.train_loader)

    def _save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        if self._dist:
            self.method.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.method.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
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

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(self.args.method, fps)
        else:
            fps = ''
        print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)

    def train(self):
        """Training loops of STL methods"""
        recorder = Recorder(verbose=True)
        num_updates = self._epoch * self.steps_per_epoch
        self.call_hook('before_train_epoch')

        eta = 1.0  # PredRNN variants
        for epoch in range(self._epoch, self._max_epochs):
            num_updates, loss_mean, eta = self.method.train_one_epoch(self, self.train_loader,
                                                                      epoch, num_updates, eta)

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)

                if self._rank == 0:
                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    recorder(vali_loss, self.method.model, self.path)
                    self._save(name='latest')
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        if self._dist:
            self.method.model.module.load_state_dict(torch.load(best_model_path))
        else:
            self.method.model.load_state_dict(torch.load(best_model_path))
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def vali(self, vali_loader):
        """A validation loop during training"""
        self.call_hook('before_val_epoch')
        preds, trues, val_loss = self.method.vali_one_epoch(self, self.vali_loader)
        self.call_hook('after_val_epoch')

        if self._rank == 0:
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
        """A testing loop of STL methods"""
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            if self._dist:
                self.method.model.module.load_state_dict(torch.load(best_model_path))
            else:
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

        if self._rank == 0:
            print_log(eval_log)
            folder_path = osp.join(self.path, 'saved')
            check_dir(folder_path)

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])

        return eval_res['mse']
