from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
import toml
import torch
import torchmetrics as tm

from mlkit.nn.base import MLKitAbstractModule


class TestBaseMLModel:
    @pytest.fixture
    def conf(self):
        conf = MagicMock()
        conf.model = MagicMock()
        conf.model.arguments = PropertyMock(
            return_value={"input_dims": 10, "output_dims": 1}
        )
        conf.metrics_obj = PropertyMock(
            return_value={
                "Precision": tm.Precision(task="binary"),
                "FBetaScore": tm.FBetaScore(task="binary"),
            }
        )

        conf.training = MagicMock()
        conf.training.optimizer = MagicMock()
        conf.training.optimizer.optimizer = PropertyMock(
            return_value=torch.optim.SGD
        )
        conf.parameters = MagicMock(return_value=None)
        yield conf

    @pytest.fixture
    def custom_module(self, conf):
        class CustomModule(MLKitAbstractModule):
            setup = MagicMock()
            step = MagicMock()

        obj = CustomModule(conf=conf)
        yield obj

    def test_setup_metric_trackers(self, custom_module):
        assert custom_module.train_metric_tracker is not None
        assert custom_module.val_metric_tracker is not None
        assert custom_module.test_metric_tracker is not None

    def test_setup_metric_stores(self, custom_module):
        assert custom_module.train_metric_tracker._metrics.return_value == {
            "Precision": tm.Precision(task="binary"),
            "FBetaScore": tm.FBetaScore(task="binary"),
        }
        assert custom_module.val_metric_tracker._metrics.return_value == {
            "Precision": tm.Precision(task="binary"),
            "FBetaScore": tm.FBetaScore(task="binary"),
        }
        assert custom_module.test_metric_tracker._metrics.return_value == {
            "Precision": tm.Precision(task="binary"),
            "FBetaScore": tm.FBetaScore(task="binary"),
        }

    def test_setup_called(self, custom_module, conf):
        custom_module.setup.assert_called_with(**conf.model.arguments)
