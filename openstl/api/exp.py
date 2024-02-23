# Copyright (c) CAIRI AI Lab. All rights reserved

import sys
import time
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch

from openstl.methods import method_maps
from openstl.datasets import BaseDataModule
from openstl.utils import (get_dataset, measure_throughput, SetupCallback, EpochEndCallback, BestCheckpointCallback)

import argparse
from pytorch_lightning import seed_everything, Trainer
import pytorch_lightning.callbacks as plc


class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None, strategy='ddp'):
        """Initialize experiments (non-dist as an example)"""
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist

        base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name if not args.ex_name.startswith(args.res_dir) \
            else args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(args.seed)
        self.data = self._get_data(dataloaders)
        self.method = method_maps[self.args.method](steps_per_epoch=len(self.data.train_loader), \
            test_mean=self.data.test_mean, test_std=self.data.test_std, save_dir=save_dir, **self.config)
        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        self.trainer = self._init_trainer(self.args, callbacks, strategy)

    def _init_trainer(self, args, callbacks, strategy):
        trainer_config = {
            'devices': args.gpus,  # Use the all GPUs
            'max_epochs': args.epoch,  # Maximum number of epochs to train for
            "strategy": strategy, # 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_false'
            'accelerator': 'gpu',  # Use distributed data parallel
            'callbacks': callbacks,
        }
        return Trainer(device=args.gpus,  # Use these GPUs
                       max_epochs=args.epoch,  # Maximum number of epochs to train for
                       strategy=strategy,   # 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_false'
                       accelerator='gpu',  # Use distributed data parallel
                       callbacks=callbacks
                    )

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._dist == 0:
            if not self.args.no_display_method_info:
                method_info = self.display_method_info(args)

        setup_callback = SetupCallback(
            prefix = 'train' if (not args.test) else 'test',
            setup_time = time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir = save_dir,
            ckpt_dir = ckpt_dir,
            args = args,
            method_info = method_info,
            argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
        )

        ckpt_callback = BestCheckpointCallback(
            monitor=args.metric_for_bestckpt,
            filename='best-{epoch:02d}-{val_loss:.3f}',
            mode='min',
            save_last=True,
            dirpath=ckpt_dir,
            verbose=True,
            every_n_epochs=args.log_step,
        )
        
        epochend_callback = EpochEndCallback()

        callbacks = [setup_callback, ckpt_callback, epochend_callback]
        if args.sched:
            callbacks.append(plc.LearningRateMonitor(logging_interval=None))
        return callbacks, save_dir

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            train_loader, vali_loader, test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            train_loader, vali_loader, test_loader = dataloaders

        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)

    def train(self):
        self.trainer.fit(self.method, self.data)

    def test(self):
        if self.args.test == True:
            ckpt = torch.load(osp.join(self.save_dir, 'checkpoints', 'best.ckpt'))
            self.method.load_state_dict(ckpt['state_dict'])
        self.trainer.test(self.method, self.data)
    
    def display_method_info(self, args):
        """Plot the basic infomation of supported methods"""
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            device = torch.device(assign_gpu)
        T, C, H, W = args.in_shape
        if args.method in ['simvp', 'tau', 'mmvp', 'wast']:
            input_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
        elif args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
            _tmp_input2 = torch.ones(1, args.aft_seq_length, C, H, W).to(device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        elif args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau']:
            Hp, Wp = H // args.patch_size, W // args.patch_size
            Cp = args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, args.total_length, Hp, Wp, Cp).to(device)
            _tmp_flag = torch.ones(1, args.aft_seq_length - 1, Hp, Wp, Cp).to(device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif args.method in ['swinlstm_d', 'swinlstm_b']:
            input_dummy = torch.ones(1, self.args.total_length, H, W, C).to(device)
        elif args.method == 'predrnnv2':
            Hp, Wp = H // args.patch_size, W // args.patch_size
            Cp = args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, args.total_length, Hp, Wp, Cp).to(device)
            _tmp_flag = torch.ones(1, args.total_length - 2, Hp, Wp, Cp).to(device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif args.method == 'prednet':
           input_dummy = torch.ones(1, 1, C, H, W, requires_grad=True).to(device)
        else:
            raise ValueError(f'Invalid method name {args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model.to(device), input_dummy)
        flops = flop_count_table(flops)
        if args.fps:
            fps = measure_throughput(self.method.model.to(device), input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line