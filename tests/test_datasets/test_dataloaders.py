"""Test data loading pipelines and dataloaders."""
import pytest

# Skip all tests in this module if lightning is not available
pytest.importorskip("lightning")

import torch
from torch.utils.data import DataLoader, Dataset

from openstl.datasets import BaseDataModule


class DummyDataset(Dataset):
    """A dummy dataset for testing."""

    def __init__(self, num_samples=100, seq_length=10, channels=1, height=64, width=64):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.channels = channels
        self.height = height
        self.width = width
        self.mean = 0.0  # Required by BaseDataModule
        self.std = 1.0   # Required by BaseDataModule
        self.data_name = "dummy"  # Required by BaseDataModule

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return (input, target) pair
        input_tensor = torch.randn(self.seq_length, self.channels, self.height, self.width)
        target_tensor = torch.randn(self.seq_length, self.channels, self.height, self.width)
        return input_tensor, target_tensor


class TestDataModule:
    """Test BaseDataModule functionality."""

    def test_datamodule_creation(self):
        """Test BaseDataModule creation."""
        # Create dummy dataloaders
        train_dataset = DummyDataset(num_samples=50)
        vali_dataset = DummyDataset(num_samples=20)
        test_dataset = DummyDataset(num_samples=20)

        train_loader = DataLoader(train_dataset, batch_size=4)
        vali_loader = DataLoader(vali_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=4)

        data_module = BaseDataModule(train_loader, vali_loader, test_loader)

        assert data_module is not None
        assert data_module.train_loader is not None
        assert data_module.valid_loader is not None  # Fixed: use valid_loader
        assert data_module.test_loader is not None

    def test_dataloader_iteration(self):
        """Test that dataloaders can be iterated."""
        train_dataset = DummyDataset(num_samples=20)
        train_loader = DataLoader(train_dataset, batch_size=4)

        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            input_tensor, target_tensor = batch
            # Fixed: input shape is (B, T, C, H, W) = 5 dimensions
            assert input_tensor.shape[0] <= 4  # batch size (last batch may be smaller)
            assert len(input_tensor.shape) == 5  # (B, T, C, H, W)

        assert batch_count == 5  # 20 samples / 4 batch size


class TestDatasetUtils:
    """Test dataset utility functions."""

    def test_ordered_distributed_sampler_import(self):
        """Test that OrderedDistributedSampler can be imported."""
        from openstl.datasets.utils import OrderedDistributedSampler, RepeatAugSampler
        assert OrderedDistributedSampler is not None
        assert RepeatAugSampler is not None
