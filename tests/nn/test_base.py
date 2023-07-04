import os
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml

import torch
import torchmetrics as tm

from mlkit.nn.base import MLKitAbstractModule
from tests.fixtures import conf


class TestBaseMLModel:
    @pytest.fixture
    def custom_module(self, conf):
        class CustomModule(MLKitAbstractModule):
            configure = MagicMock()
            run_step = MagicMock()

        obj = CustomModule(conf=conf)
        yield obj

    @pytest.fixture
    def module(self):
        from torch import nn

        class Module(MLKitAbstractModule):
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

    def test_load_from_checkpoint(self, module):
        chkpt_path = os.path.join(
            "tests",
            "resources",
            "epoch=2_val_jaccardindexmacro=0.26_cnn.ckpt",
        )
        loaded_module = module.load_from_checkpoint(chkpt_path)
        assert loaded_module._conf
        assert loaded_module._criterion
        pass
