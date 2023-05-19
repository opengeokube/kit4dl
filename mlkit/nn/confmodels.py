"""A module with configuration classes"""
import os
from abc import ABC
from functools import partial
from inspect import signature
from typing import Any, Callable, Literal

import torch
import torchmetrics as tm
from pydantic import BaseModel, Field, root_validator, validator

import mlkit.io as io_
from mlkit import utils as ut
from mlkit.nn.validators import (
    validate_class_exists,
    validate_cuda_device_exists,
)
from mlkit.types import FullyQualifiedName


# ################################
#           ABSTRACT
# ################################
class _AbstractClassWithArgumentsConf(
    ABC, BaseModel, extra="allow", allow_mutation=False
):
    target: FullyQualifiedName | str
    arguments: dict[str, Any]

    _validate_class_exists = validator("target", allow_reuse=True)(
        validate_class_exists
    )

    @root_validator(pre=True)
    def build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "arguments" in values:
            return values
        arguments = {
            k: v for k, v in values.items() if k not in cls.__fields__
        }
        values = {k: v for k, v in values.items() if k in cls.__fields__}
        values["arguments"] = arguments
        return values


# ################################
#  Basic experiment configuration
# ################################
class BaseConf(BaseModel):
    seed: int | None = Field(default=0, ge=0)
    cuda_id: int | None = None
    experiment_name: str
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"] | None = (
        "INFO"
    )

    _assert_cuda_device = validator("cuda_id", allow_reuse=True)(
        validate_cuda_device_exists
    )

    @validator("log_level", pre=True)
    def match_log_level(cls, item):
        return item.upper()

    @property
    def device(self) -> torch.device:
        if self.cuda_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.cuda_id}")

    @property
    def accelerator_device_and_id(self) -> tuple[str, int]:
        if self.cuda_id is None:
            return ("cpu", "auto")
        return ("gpu", [self.cuda_id])


# ################################
#  Neural network model configuration
# ################################
class ModelConf(_AbstractClassWithArgumentsConf):
    @validator("target")
    def check_if_target_has_expected_parent_class(cls, value):
        from mlkit.nn.base import MLKitAbstractModule

        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(
            target_class, MLKitAbstractModule
        ), f"target class must be a subclass of `{MLKitAbstractModule}` class!"
        return value

    @property
    def model_class(self) -> type:
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        return target_class


# ################################
#       Optimizer configuration
# ################################
class OptimizerConf(_AbstractClassWithArgumentsConf):
    lr: float = Field(gt=0)

    @validator("target")
    def check_if_target_has_expected_parent_class(cls, value):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, torch.optim.Optimizer), (
            "target class of the optimizer must be a subclass of"
            f" `{torch.optim.Optimizer}` class!"
        )
        return value

    @property
    def optimizer(self) -> Callable[..., torch.optim.Optimizer]:
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        return partial(target_class, lr=self.lr, **self.arguments)


# ################################
#       Checkpoint configuration
# ################################
class CheckpointConf(BaseModel):
    path: str
    monitor: dict[str, str]
    filename: str
    mode: Literal["min", "max"] | None = "max"
    save_top_k: int | None = Field(1, ge=1)
    save_weights_only: bool | None = True
    every_n_train_steps: int | None = Field(ge=1)
    save_on_train_epoch_end: bool | None = None

    @validator("path")
    def assert_path_not_exists_or_is_empty(cls, path):
        assert (
            not os.path.exists(path) or len(os.listdir(path)) == 0
        ), f"dir `{path}` exists but it's not empty!"
        return path

    @validator("monitor")
    def assert_required_monitor_keys_defined(cls, monitor):
        assert "metric" in monitor, "`metric` key is missing"
        assert (
            "stage" in monitor
        ), "`stage` key is missing. define `stage='train'` or `stage='val'`"
        assert monitor["stage"] in {"train", "val"}
        return monitor

    @property
    def monitor_metric(self) -> str:
        return "_".join(
            [self.monitor["stage"].lower(), self.monitor["metric"].lower()]
        )


# ################################
#     Criterion configuration
# ################################
class CriterionConf(BaseModel):
    target: FullyQualifiedName
    weight: list[float] | None = None

    _validate_class_exists = validator("target", allow_reuse=True)(
        validate_class_exists
    )

    @validator("target")
    def check_if_target_has_expected_parent_class(cls, value):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, torch.nn.Module), (
            "target class of the criterion must be a subclass of"
            f" `{torch.nn.Module}` class!"
        )
        return value

    @root_validator(skip_on_failure=True)
    def match_weight(cls, values):
        if values.get("weight"):
            criterion_class = (
                io_.import_and_get_attr_from_fully_qualified_name(
                    values["target"]
                )
            )
            assert "weight" in signature(criterion_class).parameters, (
                "`weight` parameter is not defined for the criterion"
                f" `{criterion_class}`"
            )
        return values

    @property
    def criterion(self) -> torch.nn.Module:
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        if self.weight is not None:
            weight_tensor = torch.FloatTensor(self.weight)
            return target_class(weight=weight_tensor)
        return target_class()


# ################################
#     Training configuration
# ################################
class TrainingConf(BaseModel):
    epochs: int = Field(gt=0)
    epoch_schedulers: list[dict[str, Any]] | None = Field(default_factory=list)
    checkpoint: CheckpointConf | None = None
    optimizer: OptimizerConf
    criterion: CriterionConf

    @validator("epoch_schedulers", each_item=True)
    def assert_epoch_scheduler_exist(cls, sch):
        assert "target" in sch
        validate_class_exists(sch["target"])
        return sch

    @property
    def preconfigured_schedulers_classes(
        self,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        schedulers = []
        for sch in self.epoch_schedulers:
            sch_copy = sch.copy()
            schedulers.append(
                partial(
                    io_.import_and_get_attr_from_fully_qualified_name(
                        sch_copy.pop("target")
                    ),
                    **sch_copy,
                )
            )
        return schedulers


# ################################
#     Validation configuration
# ################################
class ValidationConf(BaseModel):
    run_every_epoch: int = Field(gt=0)


# ################################
#     Split dataset configuration
# ################################
class SplitDatasetConf(BaseModel):
    loader: dict[str, Any] | None = Field(default_factory=dict)
    arguments: dict[str, Any]

    @root_validator(pre=True)
    def build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "arguments" in values:
            return values
        arguments = {
            k: v for k, v in values.items() if k not in cls.__fields__
        }
        values = {k: v for k, v in values.items() if k in cls.__fields__}
        values["arguments"] = arguments
        return values


# ################################
#     Dataset configuration
# ################################
class DatasetConf(BaseModel):
    target: FullyQualifiedName | str
    train: SplitDatasetConf | None = None
    validation: SplitDatasetConf | None = None
    trainval: SplitDatasetConf | None = None
    test: SplitDatasetConf | None = None
    predict: SplitDatasetConf | None = None

    @validator("target")
    def check_if_target_has_expected_parent_class(cls, value):
        from mlkit.dataset import (  # pylint: disable=import-outside-toplevel
            MLKitAbstractDataset,
        )

        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, MLKitAbstractDataset), (
            "target class of the dataset module must be a subclass of"
            f" `{MLKitAbstractDataset}` class!"
        )
        return value

    @property
    def datamodule_class(self) -> type:
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        return target_class


# ################################
#     Complete configuration
# ################################
class Conf(BaseModel):
    base: BaseConf
    model: ModelConf
    metrics: dict[str | FullyQualifiedName, dict[str, Any]] | None = Field(
        default_factory=dict
    )
    training: TrainingConf
    validation: ValidationConf
    dataset: DatasetConf

    def __init__(self, root_dir: str | None = None, **kwargs):
        if root_dir:
            kwargs = Conf.override_with_abs_target(root_dir, kwargs)
        super().__init__(**kwargs)

    @validator("metrics")
    def validate_metrics_exist(cls, values):
        if not values:
            return None
        for metric_name in values.keys():
            if "." in metric_name:
                # TODO: dot is not supported in TOML key - parsing is wrong
                raise NotImplementedError(
                    "using custom metrics is currently not defined! use"
                    " `torchmetrics` package metrics"
                )
                # TODO: logic to handle in the future
            assert hasattr(tm, metric_name), (
                f"metric `{metric_name}` is not defined in `torchmetrics`"
                " package."
            )
            _ = getattr(tm, metric_name)
        return values

    @root_validator(skip_on_failure=True)
    def check_metric_in_checkpoint_is_defined(cls, values):
        monitored_metric_name = values["training"].checkpoint.monitor["metric"]
        assert monitored_metric_name in values["metrics"].keys(), (
            f"metric `{monitored_metric_name}` is not defined. did you forget"
            f" to define `{monitored_metric_name}` in [metrics]?"
        )
        return values

    @property
    def metrics_obj(self) -> dict[str, tm.Metric]:
        if not self.metrics:
            raise ValueError("metrics are not defined!")
        return {
            metric_name.lower(): getattr(tm, metric_name)(**metric_args)
            for metric_name, metric_args in self.metrics.items()
        }

    @classmethod
    def override_with_abs_target(cls, root_dir, entries: dict) -> dict:
        replace_logic = partial(io_.maybe_get_abs_target, root_dir=root_dir)
        entries = ut.replace_item_recursively(
            entries, key="target", replace=replace_logic
        )
        return entries
