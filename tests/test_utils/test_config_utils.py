# Copyright (c) CAIRI AI Lab. All rights reserved
"""Tests for config utility functions."""

import pytest
from openstl.utils.main_utils import update_config


class TestUpdateConfig:
    """Tests for the update_config function."""

    def test_config_overrides_default_args(self):
        """Config file values should override argparse default values."""
        args = {'dataname': 'mmnist', 'batch_size': 16}
        config = {'dataname': 'custom_dataset', 'batch_size': 32}

        result = update_config(args, config)

        assert result['dataname'] == 'custom_dataset', \
            "Config value should override default CLI value"
        assert result['batch_size'] == 32, \
            "Config value should override default CLI value"

    def test_config_none_values_ignored(self):
        """None values in config should not override args."""
        args = {'dataname': 'mmnist', 'batch_size': 16}
        config = {'dataname': None, 'batch_size': 32}

        result = update_config(args, config)

        assert result['dataname'] == 'mmnist', \
            "None config value should not override arg"
        assert result['batch_size'] == 32, \
            "Non-None config value should override arg"

    def test_exclude_keys_respected(self):
        """Keys in exclude_keys should not be overridden."""
        args = {'dataname': 'mmnist', 'batch_size': 16}
        config = {'dataname': 'custom_dataset', 'batch_size': 32}

        result = update_config(args, config, exclude_keys=['dataname'])

        assert result['dataname'] == 'mmnist', \
            "Excluded key should not be overridden"
        assert result['batch_size'] == 32, \
            "Non-excluded key should be overridden"

    def test_empty_config(self):
        """Empty config should not change args."""
        args = {'dataname': 'mmnist', 'batch_size': 16}
        config = {}

        result = update_config(args, config)

        assert result['dataname'] == 'mmnist'
        assert result['batch_size'] == 16

    def test_empty_args(self):
        """Empty args should be populated from config."""
        args = {}
        config = {'dataname': 'mmnist', 'batch_size': 16}

        result = update_config(args, config)

        assert result['dataname'] == 'mmnist'
        assert result['batch_size'] == 16

    def test_new_keys_added(self):
        """New keys from config should be added to args."""
        args = {'dataname': 'mmnist'}
        config = {'dataname': 'mmnist', 'batch_size': 32, 'lr': 0.001}

        result = update_config(args, config)

        assert result['dataname'] == 'mmnist'
        assert result['batch_size'] == 32
        assert result['lr'] == 0.001
