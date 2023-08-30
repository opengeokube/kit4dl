import os
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml

import torch
import torchmetrics as tm

from kit4dl.nn.base import Kit4DLAbstractModule
from tests.fixtures import conf


class TestBaseMLModel:
    @pytest.fixture
    def custom_module(self, conf):
        class CustomModule(Kit4DLAbstractModule):
            configure = MagicMock()
            run_step = MagicMock()

        obj = CustomModule(conf=conf)
        yield obj

    @pytest.fixture
    def module(self):
        from torch import nn

        class Module(Kit4DLAbstractModule):
            def configure(self, input_dims, output_dims) -> None:
                self.l1 = nn.Sequential(
                    nn.Conv1d(
                        input_dims,
                        128,
                        kernel_size=3,
                        padding="same",
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(
                        128,
                        output_dims,
                        kernel_size=3,
                        padding="same",
                        bias=True,
                    ),
                )

            def run_step(
                self, batch, batch_idx
            ) -> tuple[torch.Tensor, torch.Tensor]:
                x, cat, instance = batch
                res = self.l1(x.permute(0, 2, 1))
                return cat, res

        return Module

    def test_setup_metric_trackers(self, custom_module):
        assert custom_module.train_metric_tracker is not None
        assert custom_module.val_metric_tracker is not None
        assert custom_module.test_metric_tracker is not None

    def test_setup_called(self, custom_module, conf):
        custom_module.configure.assert_called_with(**conf.model.arguments)
