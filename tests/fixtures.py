import os

import pytest

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml

from unittest.mock import MagicMock, Mock, PropertyMock

import torch
import torchmetrics as tm

from kit4dl.nn.confmodels import Conf


@pytest.fixture
def conf():
    conf = MagicMock()
    conf.base = PropertyMock()
    conf.base.accelerator_device_and_id = PropertyMock(
        return_value="cuda"
    ), PropertyMock(return_value=0)
    conf.logging.level = "INFO"
    conf.logging.format_ = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    conf.base.seed = 0
    conf.model = PropertyMock()
    conf.model.arguments = {"input_dims": 10, "output_dims": 1}
    conf.metrics_obj = {
        "Precision": tm.Precision(task="binary"),
        "FBetaScore": tm.FBetaScore(task="binary"),
    }
    conf.training = PropertyMock()
    conf.training.optimizer = PropertyMock()
    conf.training.optimizer.optimizer = torch.optim.SGD
    conf.training.checkpoint.mode = "max"

    conf.model.model_class = PropertyMock()

    conf.dataset = PropertyMock()
    conf.dataset.datamodule_class = PropertyMock()

    conf.dict = Mock()
    conf.dict.return_value = {}
    yield conf


@pytest.fixture
def true_conf():
    yield Conf(
        root_dir=os.path.join(os.getcwd(), "tests", "resources"),
        **toml.load(os.path.join("tests", "resources", "test_file.toml")),
    )


@pytest.fixture
def base_conf_txt():
    return """
    [base]
    seed = 0
    experiment_name = "handwritten_digit_classification"

    [model]
    target = "./tests/dummy_module.py::B"
    input_dims = 1
    layers = 4
    dropout = 0.5
    output_dims = 10

    [training]
    epochs = 100
    epoch_schedulers = [
        {target = "torch.optim.lr_scheduler::CosineAnnealingLR", T_max = 100}
    ]

    [training.checkpoint]
    path = "chckpt"
    monitor = {"metric" = "Precision", "stage" = "val"}
    filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
    mode = "max"

    [training.optimizer]
    target = "torch.optim::Adam"
    lr = 0.0001
    weight_decay = 0.03

    [training.criterion]
    target = "torch.nn::NLLLoss"
    weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    [validation]
    run_every_epoch = 1

    [dataset]
    target = "./tests/dummy_module.py::DummyDatasetModule"

    [dataset.trainval]
    root_dir = "./mnist"

    [dataset.train.loader]
    batch_size = 10
    shuffle = true
    num_workers = 4

    [dataset.validation.loader]
    batch_size = 10
    shuffle = false
    num_workers = 4
    """


@pytest.fixture
def base_conf_txt_full(base_conf_txt):
    return base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes = 10}
    """


@pytest.fixture
def dummy_optimizer():
    yield torch.optim.Adam([torch.nn.Parameter(torch.rand((100,)))], lr=0.1)
