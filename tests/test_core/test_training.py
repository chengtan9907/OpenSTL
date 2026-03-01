"""Test training loop and integration."""
import pytest

# Skip all tests in this module if lightning is not available
pytest.importorskip("lightning")

import torch
from torch.utils.data import DataLoader, Dataset

from openstl.api import BaseExperiment
from openstl.methods import method_maps
from openstl.datasets import BaseDataModule


class DummyPredDataset(Dataset):
    """A dummy dataset for prediction testing."""

    def __init__(self, num_samples=100, seq_length=10, channels=1, height=64, width=64):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.channels = channels
        self.height = height
        self.width = width
        self.mean = 0.0
        self.std = 1.0
        self.data_name = "dummy"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_tensor = torch.randn(self.seq_length, self.channels, self.height, self.width)
        target_tensor = torch.randn(self.seq_length, self.channels, self.height, self.width)
        return input_tensor, target_tensor


class TestMinimalTraining:
    """Test minimal training loop."""

    def test_simvp_one_epoch(self):
        """Test SimVP training for one epoch."""
        from openstl.methods.simvp import SimVP

        # Create dummy dataloaders
        train_dataset = DummyPredDataset(num_samples=16, seq_length=10, height=32, width=32)
        test_dataset = DummyPredDataset(num_samples=8, seq_length=10, height=32, width=32)

        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=4)

        # Create minimal config
        class Args:
            method = 'simvp'
            dataname = 'dummy'
            in_shape = (10, 1, 32, 32)
            total_length = 20
            aft_seq_length = 10
            pre_seq_length = 10
            hid_S = 16
            hid_T = 64
            N_T = 2
            N_S = 2
            epoch = 1
            lr = 1e-3
            batch_size = 4
            sched = 'cosine'
            warmup_epoch = 0
            min_lr = 1e-5
            opt = 'adam'
            weight_decay = 0.0
            seed = 42
            gpus = [0] if torch.cuda.is_available() else [0]
            dist = 0
            log_step = 1
            metric_for_bestckpt = 'val_loss'
            ex_name = 'test_simvp'
            res_dir = 'work_dirs/test'
            metrics = ['mae', 'mse']
            steps_per_epoch = len(train_loader)
            save_dir = 'work_dirs/test/test_simvp'
            test_mean = 0.0
            test_std = 1.0

        args = Args()

        # Create method directly
        method = SimVP(
            steps_per_epoch=len(train_loader),
            test_mean=0.0,
            test_std=1.0,
            save_dir='work_dirs/test/test_simvp',
            **args.__dict__
        )

        # Create data module
        data = BaseDataModule(train_loader, test_loader, test_loader)

        # Create trainer
        from lightning import Trainer
        trainer = Trainer(
            devices=1,
            max_epochs=1,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            enable_checkpointing=False,
            logger=False,
        )

        # Run training
        trainer.fit(method, data)

        assert trainer.state.finished, "Training failed"

    def test_method_initialization(self):
        """Test that all methods can be initialized."""
        for method_name, method_class in method_maps.items():
            assert method_class is not None, f"Method {method_name} is None"
