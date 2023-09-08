import importlib
import sys

import pytest

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml

import lightning.pytorch.loggers as pl_logs
import torch
import torchmetrics as tm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from pydantic import ValidationError

from kit4dl.nn.base import Kit4DLAbstractModule
from kit4dl.nn.confmodels import (
    BaseConf,
    CheckpointConf,
    Conf,
    CriterionConf,
    DatasetConf,
    LoggingConf,
    ModelConf,
    OptimizerConf,
    _AbstractClassWithArgumentsConf,
)
from tests.fixtures import (
    base_conf_txt,
    base_conf_txt_full,
    dummy_optimizer,
    true_conf,
)
from tests.utils import skipnocuda


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

    @skipnocuda
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

    @skipnocuda
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
        assert issubclass(conf.model_class, Kit4DLAbstractModule)

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

    def test_checkpoint_monitor_metric(self):
        load = """
            path = "chckpt"
            monitor = {"metric" = "Precision", "stage" = "val"}
            filename = "{epoch}_{val_fbeta_score:.2f}_convlstm"
            mode = "max"  
        """
        conf = CheckpointConf(**toml.loads(load))
        assert conf.monitor_metric_name == "val_precision"


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
        with pytest.raises(
            ValidationError,
            match=r"target class of the criterion must be a subclass of*",
        ):
            _ = CriterionConf(**toml.loads(load))


class TestDatasetConf:
    def test_get_dataset_fail_on_wrong_parent_class(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModuleWrong"
        """
        with pytest.raises(
            ValidationError,
            match=r"target class of the dataset module must be a subclass of*",
        ):
            _ = DatasetConf(**toml.loads(load))

    def test_define_trainval_and_train_loader(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModule"

            [trainval]
            root_dir = "..."

            [train.loader]
            batch_size = 100
        """
        conf = DatasetConf(**toml.loads(load))
        assert conf.trainval is not None
        assert conf.trainval.arguments == {"root_dir": "..."}
        assert conf.train is not None
        assert conf.train.loader == {"batch_size": 100}

    def test_define_val_train_test_predict_none_on_trainval_defined(self):
        load = """
            target = "tests.dummy_module::DummyDatasetModule"

            [trainval]
            root_dir = "..."
        """
        conf = DatasetConf(**toml.loads(load))
        assert conf.trainval is not None
        assert conf.train is None
        assert conf.validation is None
        assert conf.test is None
        assert conf.predict is None


class TestConf:
    @pytest.fixture
    def full_conf_dict(self, base_conf_txt):
        entries = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes=10}
        """
        yield toml.loads(entries)

    def test_conf_parse(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes = 10}
        FBetaScore = {target = "torchmetrics::Recall", task = "multiclass", num_classes = 10, beta = 0.1}        
        """
        _ = Conf(**toml.loads(load))

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="test for Python < 3.11"
    )
    def test_fail_on_duplicated_key_name(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes = 10}
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes = 10}
        """
        with pytest.raises(ValueError, match="Duplicate keys!"):
            _ = Conf(**toml.loads(load))

    @pytest.mark.skipif(
        sys.version_info > (3, 10), reason="test for Python > 3.10"
    )
    def test_fail_on_duplicated_key_name(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes=10}
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes=10}
        """
        with pytest.raises(ValueError, match="Cannot overwrite a value.*"):
            _ = Conf(**toml.loads(load))

    @pytest.mark.skip(
        reason=(
            "dots are not supported in TOML keys. cannot pass fully qualified"
            " name"
        )
    )
    def test_conf_custom_metric(self, base_conf_txt):
        load = base_conf_txt + """
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
    def test_conf_custom_metric_fail_on_wrong_parentclass(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        teset.dummy_module.CustomMetricWrong = {}
        """
        with pytest.raises(ValidationError, match="duplicate"):
            _ = Conf(**toml.loads(load))

    def test_conf_fail_on_nonexisting_metric(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        MyMetric = {target = "torchmetrics::NonExistingMetric"}
        """
        with pytest.raises(
            AttributeError,
            match=(
                r"module 'torchmetrics' has no attribute 'NonExistingMetric'"
            ),
        ):
            _ = Conf(**toml.loads(load))

    def test_conf_fail_on_monitoring_undefined_metric(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision"}
        """
        load_dict = toml.loads(load)
        load_dict["training"]["checkpoint"]["monitor"] = {
            "metric": "Recall",
            "stage": "val",
        }
        with pytest.raises(ValidationError):
            _ = Conf(**load_dict)

    def test_conf_get_metric_obj_failed_on_missing_target(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {}
        """
        with pytest.raises(
            ValidationError,
            match=r".*`target` is not defined for some metric.*",
        ):
            conf = Conf(**toml.loads(load))

    def test_conf_get_metric_obj_failed_on_missing_task(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision"}
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

    def test_conf_get_metric_obj(self, base_conf_txt):
        load = base_conf_txt + """
        [metrics]
        Precision = {target = "torchmetrics::Precision", task = "multiclass", num_classes = 10}
        """
        conf = Conf(**toml.loads(load))
        metrics = conf.metrics_obj
        assert "precision" in metrics
        assert isinstance(metrics["precision"], tm.Metric)

    def test_conf_schedulers_single_preconfigured_schedulers_classes(
        self, full_conf_dict, dummy_optimizer
    ):
        conf = Conf(**full_conf_dict)
        assert isinstance(conf.training.preconfigured_schedulers_classes, list)

    def test_conf_schedulers_preconfigured_schedulers_classes(
        self, full_conf_dict, dummy_optimizer
    ):
        conf = Conf(**full_conf_dict)
        scheduler = conf.training.preconfigured_schedulers_classes[0](
            dummy_optimizer
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

    def test_conf_schedulers_double_preconfigured_schedulers_classes(
        self, full_conf_dict, dummy_optimizer
    ):
        conf = Conf(**full_conf_dict)
        assert len(conf.training.preconfigured_schedulers_classes) == 1
        assert isinstance(
            conf.training.preconfigured_schedulers_classes[0](dummy_optimizer),
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )

        full_conf_dict["training"]["epoch_schedulers"].append(
            {
                "target": "torch.optim.lr_scheduler::MultiStepLR",
                "milestones": [30, 80],
                "gamma": 0.1,
            }
        )
        conf = Conf(**full_conf_dict)
        assert len(conf.training.preconfigured_schedulers_classes) == 2
        assert isinstance(
            conf.training.preconfigured_schedulers_classes[0](dummy_optimizer),
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )
        assert isinstance(
            conf.training.preconfigured_schedulers_classes[1](dummy_optimizer),
            torch.optim.lr_scheduler.MultiStepLR,
        )

    def test_use_base_exp_name_for_metric_logging(self, base_conf_txt_full):
        load = base_conf_txt_full + """
        [logging]
        type = "csv"
        """
        conf = Conf(**toml.loads(load))
        assert "name" in conf.logging.arguments
        assert (
            conf.logging.arguments["name"]
            == "handwritten_digit_classification"
        )

    def test_dont_override_exp_name_with_base_if_provided(
        self, base_conf_txt_full
    ):
        load = base_conf_txt_full + """
        [logging]
        type = "csv"
        name = "logging_exp_name"
        """
        conf = Conf(**toml.loads(load))
        assert "name" in conf.logging.arguments
        assert conf.logging.arguments["name"] == "logging_exp_name"


class TestLogging:
    @pytest.mark.parametrize(
        "lvl", ["debug", "warn", "info", "critical", "error"]
    )
    def test_get_log_lowercase(self, lvl):
        load = f"""
    level = "{lvl}"
        """
        assert LoggingConf(**toml.loads(load)).level == lvl.upper()

    def test_get_log_fail_on_wrong_lvl(self):
        load = f"""
    level = "not-existing"
        """
        with pytest.raises(ValidationError, match=r""):
            _ = LoggingConf(**toml.loads(load)).level

    @pytest.mark.parametrize(
        "metric_logger_nick",
        ["comet", "csv", "mlflow", "neptune", "tensorboard", "wandb"],
    )
    def testmetric_logger_type_available(self, metric_logger_nick):
        load = f"""
            type = "{metric_logger_nick}"
        """
        LoggingConf(**toml.loads(load))

    def test_fail_on_wrongmetric_logger_type(self):
        load = """
            type = "not_supported"
        """
        with pytest.raises(
            ValidationError,
            match=(
                ".*Input should be 'comet', 'csv', 'mlflow', 'neptune',"
                " 'tensorboard' or 'wandb'.*"
            ),
        ):
            LoggingConf(**toml.loads(load))

    def test_provide_kwargs_for_metric_logger(self):
        load = """
            type = "csv"
            arg1 = 1
            arg2 = "2"
        """
        conf = LoggingConf(**toml.loads(load))
        assert conf.arguments.get("arg1") == 1
        assert conf.arguments.get("arg2") == "2"

    @pytest.mark.parametrize(
        "log_nick, log_class",
        [
            ("comet", pl_logs.CometLogger),
            ("csv", pl_logs.CSVLogger),
            ("mlflow", pl_logs.MLFlowLogger),
            ("neptune", pl_logs.NeptuneLogger),
            ("tensorboard", pl_logs.TensorBoardLogger),
            ("wandb", pl_logs.WandbLogger),
        ],
    )
    def test_metric_logger_class(self, log_nick, log_class):
        load = f"""
            type = "{log_nick}"
        """
        conf = LoggingConf(**toml.loads(load))
        assert conf.metric_logger_type == log_class

    def test_default_on_empty_string(self):
        load = ""
        conf = LoggingConf(**toml.loads(load))
        assert conf.level
        assert conf.format_
        assert conf.type_

    @pytest.mark.parametrize(
        "log_nick, attr_name",
        [
            ("comet", "experiment_name"),
            ("csv", "name"),
            ("mlflow", "experiment_name"),
            ("neptune", "name"),
            ("tensorboard", "name"),
            ("wandb", "name"),
        ],
    )
    def test_update_project_name_if_undefined(self, log_nick, attr_name):
        EXP_NAME = "new_exp_name"
        load = f"""
            type = "{log_nick}"
        """
        conf = LoggingConf(**toml.loads(load))
        conf.maybe_update_experiment_name(EXP_NAME)
        assert conf.arguments[attr_name] == EXP_NAME

    @pytest.mark.parametrize(
        "log_nick, attr_name",
        [
            ("comet", "experiment_name"),
            ("csv", "name"),
            ("mlflow", "experiment_name"),
            ("neptune", "name"),
            ("tensorboard", "name"),
            ("wandb", "name"),
        ],
    )
    def test_does_not_override_project_name_if_defined(
        self, log_nick, attr_name
    ):
        EXP_NAME = "another_new_exp"
        OVERRIDE_EXP_NAME = "new_exp_name"
        load = f"""
            type = "{log_nick}"
            {attr_name} = "{EXP_NAME}"
        """
        conf = LoggingConf(**toml.loads(load))
        conf.maybe_update_experiment_name(OVERRIDE_EXP_NAME)
        assert conf.arguments[attr_name] == EXP_NAME

    def test_properly_parse_attributes(self):
        load = """
        type = "csv"
        level = "info"
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        """
        conf = LoggingConf(**toml.loads(load))
        assert conf.type_ == "csv"
        assert conf.level == "INFO"
        assert (
            conf.format_
            == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
