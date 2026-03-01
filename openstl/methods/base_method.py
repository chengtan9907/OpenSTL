import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams,
            self.hparams.epoch,
            self.model,
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError

    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)

        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list,
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))

        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            # Save original numpy files
            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

            # Save structured JSON output
            json_output = {
                'metrics': {
                    'mae': float(eval_res['mae']),
                    'mse': float(eval_res['mse']),
                },
                'config': dict(self.hparams),
            }
            # Add per-channel metrics if available
            if 'per_channel' in eval_res:
                json_output['metrics']['per_channel'] = eval_res['per_channel']

            with open(osp.join(folder_path, 'results.json'), 'w') as f:
                json.dump(json_output, f, indent=2)

            # Save structured CSV output
            import csv
            csv_path = osp.join(folder_path, 'results.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['mae', f"{eval_res['mae']:.6f}"])
                writer.writerow(['mse', f"{eval_res['mse']:.6f}"])
                if 'per_channel' in eval_res:
                    for ch_name, ch_metrics in eval_res['per_channel'].items():
                        for metric_name, metric_value in ch_metrics.items():
                            writer.writerow([f'{ch_name}_{metric_name}', f'{metric_value:.6f}'])

        # Clear test_outputs for next test run
        self.test_outputs.clear()
        return results_all
