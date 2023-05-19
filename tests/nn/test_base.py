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
        conf.base = MagicMock()
        conf.base.log_level = "INFO"
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
            configure = MagicMock()
            step = MagicMock()

        obj = CustomModule(conf=conf)
        yield obj

    def test_setup_metric_trackers(self, custom_module):
        assert custom_module.train_metric_tracker is not None
        assert custom_module.val_metric_tracker is not None
        assert custom_module.test_metric_tracker is not None

    def test_setup_called(self, custom_module, conf):
        custom_module.configure.assert_called_with(**conf.model.arguments)

    @pytest.mark.parametrize(
        "log_method", ["debug", "info", "warn", "error", "critical"]
    )
    def test_logging_methods(self, custom_module, log_method):
        assert hasattr(custom_module, log_method)
        assert callable(getattr(custom_module, log_method))

    def test_log_methods_use_configured_handler(self):
        # TODO:
        pass
