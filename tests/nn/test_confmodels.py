import importlib

import pytest
import toml
import torch
import torchmetrics as tm
from pydantic import ValidationError

from mlkit.nn.base import MLKitAbstractModule
from mlkit.nn.confmodels import (
    BaseConf,
    CheckpointConf,
    Conf,
    CriterionConf,
    DatasetConfig,
    ModelConf,
    OptimizerConf,
    _AbstractClassWithArgumentsConf,
)
from tests.test_utils import skipnocuda


class TestBaseConfAndAccessor:
    @skipnocuda
    def test_base_conf_all_passed(self):
        load = """
    seed = 0
    cuda_id = 0
    experiment_name = "handwritten_digit_classification"
        """
        conf = BaseConf(**toml.loads(load))
        assert conf.seed == 0
        assert conf.cuda_id == 0
        assert conf.experiment_name == "handwritten_digit_classification"

    def test_base_conf_no_cuda_id_passed(self):
        load = """
    seed = 0
    experiment_name = "handwritten_digit_classification"
        """
        conf = BaseConf(**toml.loads(load))
        assert conf.seed == 0
        assert conf.cuda_id is None
        assert conf.experiment_name == "handwritten_digit_classification"

    def test_base_conf_no_seed_passed(self):
        load = """
    experiment_name = "handwritten_digit_classification"
        """
        conf = BaseConf(**toml.loads(load))
        assert conf.seed == 0
        assert conf.cuda_id is None
        assert conf.experiment_name == "handwritten_digit_classification"

    def test_base_conf_failed_on_missing_exp_name(self):
        load = """
    seed = 10
    cuda_id = 1
        """
        with pytest.raises(ValidationError):
            _ = BaseConf(**toml.loads(load))

    def test_base_conf_failed_on_wrong_seed(self):
        load = """
    seed_id = -10
        """
        with pytest.raises(ValidationError):
            _ = BaseConf(**toml.loads(load))

    def test_base_conf_failed_on_wrong_cuda_id(self):
        load = """
    cuda_id = -1
        """
        with pytest.raises(ValidationError):
            _ = BaseConf(**toml.loads(load))

        load = """
    cuda_id = 10
        """
        with pytest.raises(ValidationError):
            _ = BaseConf(**toml.loads(load))

    @skipnocuda
    def test_get_cuda_device_with_id_0(self):
        load = """
    cuda_id = 0
    experiment_name = "handwritten_digit_classification"
        """
        dev = BaseConf(**toml.loads(load)).device
        assert dev.type == "cuda"
        assert dev.index == 0

    def test_get_cpu_device(self):
        load = """
    experiment_name = "handwritten_digit_classification"
        """
        dev = BaseConf(**toml.loads(load)).device
        assert dev.type == "cpu"
        assert dev.index is None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="`torch` module is not installed",
)
class TestBaseClassWithArgumentsConf:
    def test_installed_class_exists(self):
        load = """
            target = "torch.optim::SGD"
        """
        conf = _AbstractClassWithArgumentsConf(**toml.loads(load))
        assert conf.target == "torch.optim::SGD"

    def test_customed_class_exists_on_path(self):
        load = """
            target = "tests.dummy_module::A"
        """
        conf = _AbstractClassWithArgumentsConf(**toml.loads(load))
        assert conf.target == "tests.dummy_module::A"

    def test_class_with_arguments(self):
        load = """
            target = "torch.optim::SGD"
            arg1 = "val1"
            arg2 = -1.5
        """
        conf = _AbstractClassWithArgumentsConf(**toml.loads(load))
        assert "arg1" in conf.arguments
        assert conf.arguments["arg1"] == "val1"
        assert "arg2" in conf.arguments
        assert conf.arguments["arg2"] == -1.5


class TestModelConf:
    def test_get_model_proper_parent_class(self):
        load = """
            target = "tests.dummy_module::B"
            input_dims = 1
            layers = 4
            dropout = 0.1
            output_dims = 10
        """
        conf = ModelConf(**toml.loads(load))
        assert issubclass(conf.model_class, MLKitAbstractModule)

    def test_get_model_failed_on_wrong_parent_class(self):
        load = """
            target = "tests.dummy_module::A"
        """
        with pytest.raises(ValidationError):
            _ = ModelConf(**toml.loads(load))


class TestOptimizerConf:
    def test_get_optimizer_proper_parent_class(self):
        load = """
            target = "torch.optim::SGD"
            lr = 0.1
        """
        conf = OptimizerConf(**toml.loads(load))
        assert callable(conf.optimizer)

        class _A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)

        optimizer = conf.optimizer(_A().parameters())
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_get_optimizer_failed_on_missing_lr(self):
        load = """
            target = "torch.optim::SGD"
        """
        with pytest.raises(ValidationError):
            _ = OptimizerConf(**toml.loads(load))

    def test_get_optimizer_failed_on_wrong_parent_class(self):
        load = """
            target = "tests.dummy_module::A"
            lr = 0.1
        """
        with pytest.raises(ValidationError):
            _ = OptimizerConf(**toml.loads(load))


class TestCheckpointConf:
    def test_checkpoint_conf(self):
        load = """
            path = "chckpt"
            monitor = {"metric" = "Precision", "stage" = "val"}
            filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
            mode = "max"  
        """
        conf = CheckpointConf(**toml.loads(load))
        assert conf.path == "chckpt"
        assert conf.monitor == {"metric": "Precision", "stage": "val"}
        assert conf.filename == "{epoch}_{val_fbeta_score:.2f}_convlstm"
        assert conf.mode == "max"

    def test_checkpoint_missing_metric_monitor(self):
        load = """
            path = "chckpt"
            monitor = {"stage" = "val"}
            filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
            mode = "max"  
        """
        with pytest.raises(ValidationError):
            _ = CheckpointConf(**toml.loads(load))

    def test_checkpoint_missing_stage_monitor(self):
        load = """
            path = "chckpt"
            monitor = {"metric" = "Precision"}
            filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
            mode = "max"  
        """
        with pytest.raises(ValidationError):
            _ = CheckpointConf(**toml.loads(load))

    def test_checkpoint_wrong_mode(self):
        load = """
            path = "chckpt"
            monitor = {"metric" = "Precision", "stage" = "val"}
            filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
            mode = "average"  
        """
        with pytest.raises(ValidationError):
            _ = CheckpointConf(**toml.loads(load))

        load = """
            path = "chckpt"
            monitor = {"metric" = "Precision", "stage" = "val"}
            filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
            mode = 0
        """
        with pytest.raises(ValidationError):
            _ = CheckpointConf(**toml.loads(load))


class TestCriterionConfig:
    def test_criterion(self):
        load = """
            target = "torch.nn::NLLLoss"
        """
        conf = CriterionConf(**toml.loads(load))
        conf.target == "torch.nn.NLLLoss"

    def test_criterion_failed_on_weight_passed_if_not_supported(self):
        load = """
            target = "torch.nn::MSELoss"
            weight = [0.1, 0.1]
        """
        with pytest.raises(ValidationError):
            _ = CriterionConf(**toml.loads(load))

    def test_criterion_weight_passed(self):
        load = """
            target = "torch.nn::NLLLoss"
            weight = [0.1, 0.1]
        """
        conf = CriterionConf(**toml.loads(load))
        conf.target == "torch.nn.NLLLoss"
        assert conf.weight == [0.1, 0.1]

    @pytest.mark.skip(reason="weights values are not validated anymore")
    def test_criterion_failed_on_negative_weight(self):
        load = """
            target = "torch.nn::NLLLoss"
            weight = [-0.1, 0.1]
        """
        with pytest.raises(ValidationError):
            _ = CriterionConf(**toml.loads(load))

    def test_get_criterion_proper_parent_class(self):
        load = """
            target = "torch.nn::MSELoss"
        """
        conf = CriterionConf(**toml.loads(load))
        assert issubclass(type(conf.criterion), torch.nn.Module)

    def test_get_criterion_failed_on_wrong_parent_class(self):
        load = """
            target = "tests.dummy_module::A"
        """
        with pytest.raises(ValidationError):
            _ = CriterionConf(**toml.loads(load))


class TestDatasetConf:
    def test_get_dataset_fail_on_wrong_parent_class(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModuleWrong"
        """
        with pytest.raises(ValidationError):
            _ = DatasetConfig(**toml.loads(load))

    def test_get_dataset_default_args(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModule"
            root_dir = "/data/mnist/train"
        """
        conf = DatasetConfig(**toml.loads(load))
        assert conf.batch_size == 1
        assert conf.shuffle is False
        assert conf.num_workers == 1
        assert conf.dataset_kwargs == {"root_dir": "/data/mnist/train"}

    def test_get_dataset_check_arguments_values(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModule"
            batch_size = 10
            shuffle = true
            num_workers = 4
            root_dir = "/data/mnist/train"
        """
        conf = DatasetConfig(**toml.loads(load))
        assert conf.batch_size == 10
        assert conf.shuffle is True
        assert conf.num_workers == 4
        assert conf.dataset_kwargs == {"root_dir": "/data/mnist/train"}

    def test_get_dataset_check_extra_arguments(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModule"
            batch_size = 10
            shuffle = true
            num_workers = 4
            root_dir = "/data/mnist/train"
            extra_arg = -10
        """
        conf = DatasetConfig(**toml.loads(load))
        assert conf.dataset_kwargs == {
            "root_dir": "/data/mnist/train",
            "extra_arg": -10,
        }


class TestConf:
    @pytest.fixture
    def base_conf(self):
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
            {target = "torch.optim.schedulers::CosineAnnealing"}
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

        [training.dataset]
        target = "./tests/dummy_module.py::DummyDatasetModule"
        batch_size = 10
        shuffle = true
        num_workers = 4
        root_dir = "/data/mnist/train"

        [validation]
        run_every_epoch = 1

        [validation.dataset]
        target = "tests.dummy_module::DummyDatasetModule"
        root_dir = "/data/mnist/val"
        batch_size = 10
        shuffle = false
        num_workers = 4

        """

    def test_conf_parse(self, base_conf):
        load = base_conf + """
        [metrics]
        Precision = {task = "multiclass", num_classes=10}
        FBetaScore = {task = "multiclass", num_classes=10, beta = 2}        
        """
        conf = Conf(**toml.loads(load))

    @pytest.mark.skip(
        reason=(
            "dots are not supported in TOML keys. cannot pass fully qualified"
            " name"
        )
    )
    def test_conf_custom_metric(self, base_conf):
        load = base_conf + """
        [metrics]
        test.dummy_module.CustomMetric = {}
        """
        conf = Conf(**toml.loads(load))

    @pytest.mark.skip(
        reason=(
            "dots are not supported in TOML keys. cannot pass fully qualified"
            " name"
        )
    )
    def test_conf_custom_metric_fail_on_wrong_parentclass(self, base_conf):
        load = base_conf + """
        [metrics]
        teset.dummy_module.CustomMetricWrong = {}
        """
        with pytest.raises(ValidationError):
            _ = Conf(**toml.loads(load))

    def test_conf_fail_on_nonexisting_metric(self, base_conf):
        load = base_conf + """
        [metrics]
        NonExistingMetric = {}
        """
        with pytest.raises(ValidationError):
            _ = Conf(**toml.loads(load))

    def test_conf_fail_on_monitoring_undefined_metric(self, base_conf):
        load = base_conf + """
        [metrics]
        Precision = {}
        """
        load_dict = toml.loads(load)
        load_dict["training"]["checkpoint"]["monitor"] = {
            "metric": "Recall",
            "stage": "val",
        }
        with pytest.raises(ValidationError):
            _ = Conf(**load_dict)

    def test_conf_get_metric_obj_failed_on_missing_task(self, base_conf):
        load = base_conf + """
        [metrics]
        Precision = {}
        """
        conf = Conf(**toml.loads(load))
        with pytest.raises(
            TypeError,
            match=(
                r"Precision.\_\_new\_\_\(\) missing 1 required positional"
                r" argument*"
            ),
        ):
            conf.metrics_obj

    def test_conf_get_metric_obj(self, base_conf):
        load = base_conf + """
        [metrics]
        Precision = {task = "multiclass", num_classes = 10}
        """
        conf = Conf(**toml.loads(load))
        metrics = conf.metrics_obj
        assert "Precision" in metrics
        assert isinstance(metrics["Precision"], tm.Metric)
