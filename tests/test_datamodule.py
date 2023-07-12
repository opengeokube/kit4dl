from unittest.mock import MagicMock

import pytest

from mlkit.dataset import MLKitAbstractDataModule


class TestDatamodule:
    class TestDataModule(MLKitAbstractDataModule):
        pass

    def test_setting_extra_arguments(self):
        conf = MagicMock()
        conf.arguments = {"abc": 1}
        datamodule = TestDatamodule.TestDataModule(conf)
        assert hasattr(datamodule, "abc")
        assert datamodule.abc == 1
