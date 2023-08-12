from unittest.mock import MagicMixin, patch

import pytest

from kit4dl.nn.trainer import Trainer, set_seed
from tests.fixtures import conf, true_conf


class TestTrainer:
    @patch("kit4dl.nn.trainer.set_seed")
    def test_seed_set_run(self, mock_set_seed, conf):
        Trainer(conf)
        mock_set_seed.assert_called_with(conf.base.seed)

    def test_seed_set_deterministic_rand_values(self, conf):
        import numpy as np
        import torch

        Trainer(conf)
        arr1 = np.random.rand(1000)
        ten1 = torch.rand(10, 20, 30)

        Trainer(conf)
        arr2 = np.random.rand(1000)
        ten2 = torch.rand(10, 20, 30)
        assert np.allclose(arr1, arr2)
        assert torch.allclose(ten1, ten2)

    @patch("kit4dl.nn.trainer.Trainer._configure_trainer")
    @patch("kit4dl.nn.trainer.Trainer._configure_datamodule")
    def test_configure_model(
        self, mock_conf_trainer, mock_conf_datamodule, conf
    ):
        Trainer(conf).prepare()
        conf.model.model_class.assert_called_once_with(conf=conf)

    @patch("kit4dl.nn.trainer.Trainer._configure_trainer")
    @patch("kit4dl.nn.trainer.Trainer._configure_model")
    def test_configure_datamodule(
        self, mock_conf_trainer, mock_conf_model, conf
    ):
        Trainer(conf).prepare()
        conf.dataset.datamodule_class.assert_called_once_with(
            conf=conf.dataset
        )
