import importlib

import pytest

import toml
from src.nn.confmodels import (
    BaseConf,
    CheckpointConf,
    CriterionConf,
    _AbstractClassWithArgumentsConf,
    BaseConfAccessor,
)
from pydantic import ValidationError

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
        import torch

        load = """
    cuda_id = 0
    experiment_name = "handwritten_digit_classification"
        """
        acc = BaseConfAccessor(BaseConf(**toml.loads(load)))
        assert acc.device.type == "cuda"
        assert acc.device.index == 0

    def test_get_cpu_device(self):
        import torch

        load = """
    experiment_name = "handwritten_digit_classification"
        """
        acc = BaseConfAccessor(BaseConf(**toml.loads(load)))
        assert acc.device.type == "cpu"
        assert acc.device.index is None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="`torch` module is not installed",
)
class TestBaseClassWithArgumentsConf:
    def test_installed_class_exists(self):
        load = """
            target = "torch.optim.SGD"
        """
        conf = _AbstractClassWithArgumentsConf(**toml.loads(load))
        assert conf.target == "torch.optim.SGD"

    def test_customed_class_exists_on_path(self):
        load = """
            target = "tests.dummy_module.A"
        """
        conf = _AbstractClassWithArgumentsConf(**toml.loads(load))
        assert conf.target == "tests.dummy_module.A"

    def test_class_with_arguments(self):
        load = """
            target = "torch.optim.SGD"
            arg1 = "val1"
            arg2 = -1.5
        """
        conf = _AbstractClassWithArgumentsConf(**toml.loads(load))
        assert "arg1" in conf.arguments
        assert conf.arguments["arg1"] == "val1"
        assert "arg2" in conf.arguments
        assert conf.arguments["arg2"] == -1.5


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
            target = "torch.nn.NLLLoss"
        """
        conf = CriterionConf(**toml.loads(load))
        conf.target == "torch.nn.NLLLoss"

    def test_criterion_weight_passed(self):
        load = """
            target = "torch.nn.NLLLoss"
            weight = [0.1, 0.1]
        """
        conf = CriterionConf(**toml.loads(load))
        conf.target == "torch.nn.NLLLoss"
        assert conf.weight == [0.1, 0.1]

    def test_criterion_failed_on_negative_weight(self):
        load = """
            target = "torch.nn.NLLLoss"
            weight = [-0.1, 0.1]
        """
        with pytest.raises(ValidationError):
            _ = CriterionConf(**toml.loads(load))
