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

    def test_setup_metric_trackers(self, custom_module):
        assert custom_module.train_metric_tracker is not None
        assert custom_module.val_metric_tracker is not None
        assert custom_module.test_metric_tracker is not None

    def test_setup_called(self, custom_module, conf):
        custom_module.configure.assert_called_with(**conf.model.arguments)
