"""Test optimizers and schedulers."""
import pytest
import torch
import torch.nn as nn

from openstl.core.optim_scheduler import (
    Adafactor, AdamP, Lookahead, Nadam,
    RAdam, RMSpropTF, NvNovoGrad, SGDP,
    CosineLRScheduler, MultiStepLRScheduler, StepLRScheduler, TanhLRScheduler
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestOptimizers:
    """Test optimizer creation and basic usage."""

    def test_adamp_optimizer(self):
        """Test AdamP optimizer."""
        model = SimpleModel()
        optimizer = AdamP(model.parameters(), lr=0.001)
        assert optimizer is not None

        # Test one step
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_adafactor_optimizer(self):
        """Test Adafactor optimizer."""
        model = SimpleModel()
        optimizer = Adafactor(model.parameters(), lr=0.001)
        assert optimizer is not None

    def test_nadam_optimizer(self):
        """Test Nadam optimizer."""
        model = SimpleModel()
        optimizer = Nadam(model.parameters(), lr=0.001)
        assert optimizer is not None

    def test_radam_optimizer(self):
        """Test RAdam optimizer."""
        model = SimpleModel()
        optimizer = RAdam(model.parameters(), lr=0.001)
        assert optimizer is not None

    def test_lookahead_wrapper(self):
        """Test Lookahead optimizer wrapper."""
        model = SimpleModel()
        base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = Lookahead(base_optimizer)
        assert optimizer is not None


class TestSchedulers:
    """Test learning rate schedulers."""

    def test_cosine_scheduler(self):
        """Test CosineLRScheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            lr_min=1e-5,
            warmup_t=2,
            warmup_lr_init=1e-6
        )

        assert scheduler is not None

        # Test stepping
        for epoch in range(5):
            scheduler.step(epoch=epoch)

    def test_step_scheduler(self):
        """Test StepLRScheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = StepLRScheduler(
            optimizer,
            decay_t=10,
            decay_rate=0.1,
            warmup_t=2
        )

        assert scheduler is not None

        for epoch in range(5):
            scheduler.step(epoch=epoch)

    def test_multistep_scheduler(self):
        """Test MultiStepLRScheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=[10, 20],
            decay_rate=0.1,
            warmup_t=2
        )

        assert scheduler is not None

        for epoch in range(5):
            scheduler.step(epoch=epoch)


class TestGetOptimScheduler:
    """Test the get_optim_scheduler function."""

    def test_get_optim_scheduler_cosine(self):
        """Test get_optim_scheduler with cosine scheduler."""
        from openstl.core.optim_scheduler import get_optim_scheduler

        class Args:
            opt = 'adam'
            lr = 0.001
            weight_decay = 0.0
            filter_bias_and_bn = True
            sched = 'cosine'
            epoch = 10
            warmup_epoch = 2
            warmup_lr = 1e-6
            min_lr = 1e-5

        args = Args()
        model = SimpleModel()

        optimizer, scheduler, by_epoch = get_optim_scheduler(args, args.epoch, model, 100)

        assert optimizer is not None
        assert scheduler is not None
