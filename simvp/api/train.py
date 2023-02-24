# Copyright (c) CAIRI AI Lab. All rights reserved

import time
import logging
import pickle
import json
import torch
import numpy as np
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

from simvp.core import metric, Recorder
from simvp.methods import method_maps
from simvp.utils import (set_seed, print_log, output_namespace, check_dir,
                         get_dataset, measure_throughput)

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

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.vali_loader, self.test_loader = get_dataset(self.args.dataname, self.config)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        recorder = Recorder(verbose=True)
        num_updates = 0
        # constants for other methods:
        eta = 1.0  # PredRNN
        for epoch in range(self.config['epoch']):
            loss_mean = 0.0

            if self.args.method in ['simvp', 'crevnet', 'phydnet']:
                num_updates, loss_mean = self.method.train_one_epoch(
                    self.train_loader, epoch, num_updates, loss_mean)
            elif self.args.method in ['convlstm', 'predrnnpp', 'predrnn', 'predrnnv2', 'mim', 'e3dlstm', 'mau']:
                num_updates, loss_mean, eta = self.method.train_one_epoch(
                    self.train_loader, epoch, num_updates, loss_mean, eta)
            else:
                raise ValueError(f'Invalid method name {self.args.method}')

            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)

                print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                    epoch + 1, len(self.train_loader), cur_lr, loss_mean, vali_loss))
                recorder(vali_loss, self.method.model, self.path)

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))

    def vali(self, vali_loader):
        preds, trues, val_loss = self.method.vali_one_epoch(self.vali_loader)

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
        inputs, trues, preds = self.method.test_one_epoch(self.test_loader)
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
