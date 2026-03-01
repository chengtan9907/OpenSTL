import json
import shutil
import logging
import os.path as osp
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from .main_utils import check_dir, collect_env, print_log, output_namespace


class SetupCallback(Callback):
    def __init__(self, prefix, setup_time, save_dir, ckpt_dir, args, method_info, argv_content=None):
        super().__init__()
        self.prefix = prefix
        self.setup_time = setup_time
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.config = args.__dict__
        self.argv_content = argv_content
        self.method_info = method_info

    def on_fit_start(self, trainer, pl_module):
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'

        if trainer.global_rank == 0:
            # check dirs
            self.save_dir = check_dir(self.save_dir)
            self.ckpt_dir = check_dir(self.ckpt_dir)
            # setup log
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO,
                filename=osp.join(self.save_dir, '{}_{}.log'.format(self.prefix, self.setup_time)),
                filemode='a', format='%(asctime)s - %(message)s')
            # print env info
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
            sv_param = osp.join(self.save_dir, 'model_param.json')
            with open(sv_param, 'w') as file_obj:
                json.dump(self.config, file_obj)

            print_log(output_namespace(self.args))
            if self.method_info is not None:
                info, flops, fps, dash_line = self.method_info
                print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)


class EpochEndCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        avg_train_loss = trainer.callback_metrics.get('train_loss', None)
        if avg_train_loss is not None:
            self.train_losses.append(float(avg_train_loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        avg_val_loss = trainer.callback_metrics.get('val_loss')

        if hasattr(self, 'train_losses') and len(self.train_losses) > 0:
            train_loss = self.train_losses[-1]
            self.val_losses.append(float(avg_val_loss) if avg_val_loss is not None else 0.0)
            print_log(f"Epoch {trainer.current_epoch}: Lr: {lr:.7f} | Train Loss: {train_loss:.7f} | Vali Loss: {avg_val_loss:.7f}")

    def on_fit_end(self, trainer, pl_module):
        """Save training history to CSV after training completes."""
        if trainer.is_global_zero and hasattr(self, 'train_losses'):
            import csv
            import os.path as osp

            # Get save_dir from trainer
            save_dir = None
            for callback in trainer.callbacks:
                if isinstance(callback, SetupCallback):
                    save_dir = callback.save_dir
                    break

            if save_dir:
                history_path = osp.join(save_dir, 'training_history.csv')
                with open(history_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'train_loss', 'val_loss'])
                    for i, (train_l, val_l) in enumerate(zip(self.train_losses, self.val_losses)):
                        writer.writerow([i, train_l, val_l])
                print_log(f"Training history saved to {history_path}")

class BestCheckpointCallback(ModelCheckpoint):
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))