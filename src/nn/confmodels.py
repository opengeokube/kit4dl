from abc import ABC
import os
from typing import Any, Literal

import torch
from pydantic import (
    BaseModel,
    Field,
    validator,
    root_validator,
    conint,
    confloat,
)

from src.typing import FullyQualifiedName
from src.nn.validators import (
    validate_cuda_device_exists,
    validate_class_exists,
)


# ################################
#           ABSTRACT
# ################################
class _AbstractAccessor(ABC):
    conf: BaseModel

    def __init__(self, conf: BaseModel) -> None:
        super().__init__()
        self.conf = conf


class _AbstractClassWithArgumentsConf(ABC, BaseModel, extra="allow"):
    target: FullyQualifiedName
    arguments: dict[str, Any] | None

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
    seed: conint(ge=0) | None = 0
    cuda_id: conint(ge=0) | None = None
    experiment_name: str

    _assert_cuda_device = validator("cuda_id", allow_reuse=True)(
        validate_cuda_device_exists
    )


class BaseConfAccessor(_AbstractAccessor):
    @property
    def device(self) -> torch.device:
        if self.conf.cuda_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.conf.cuda_id}")


# ################################
#  Neural network model configuration
# ################################
class ModelConf(_AbstractClassWithArgumentsConf):
    pass


class ModelConfAccessor(ModelConf):
    pass


# ################################
#       Optimizer configuration
# ################################
class OptimizerConf(_AbstractClassWithArgumentsConf):
    pass


class OptimizerConfAccessor(OptimizerConf):
    pass


# ################################
#       Checkpoint configuration
# ################################
class CheckpointConf(BaseModel):
    path: str
    monitor: dict[str, str]
    filename: str
    mode: Literal["min", "max"] | None = "max"

    @validator("path")
    def assert_path_not_exists_or_is_empty(cls, path):
        assert (
            not os.path.exists(path) or len(os.listdir(path)) == 0
        ), f"dir `{path}` exists but it's not empty!"
        return path

    @validator("monitor")
    def assert_required_monitor_keys_defined(cls, monitor):
        assert "metric" in monitor, f"`metric` key is missing"
        assert (
            "stage" in monitor
        ), f"`stage` key is missing. define `stage='train'` or `stage='val'`"
        return monitor


class CheckpointConfAccessor(CheckpointConf):
    pass


# ################################
#     Criterion configuration
# ################################
class CriterionConf(BaseModel):
    target: FullyQualifiedName
    weight: list[confloat(gt=0)] | None = None

    _validate_class_exists = validator("target", allow_reuse=True)(
        validate_class_exists
    )


class CriterionConfAccessor(CriterionConf):
    pass


# ################################
#       Dataset configuration
# ################################
class DatasetConfig(BaseModel):
    target: FullyQualifiedName
    batch_size: conint(gt=0)
    shuffle: bool | None = True
    num_workers: conint(gt=0) | None = 1
    dataset_kwargs: dict[str, Any] | None = Field(default_factory=dict)

    _validate_class_exists = validator("target", allow_reuse=True)(
        validate_class_exists
    )


class DatasetConfigAccessor(DatasetConfig):
    pass


# ################################
#     Training configuration
# ################################
class TrainingConf(BaseModel):
    epochs: conint(ge=1)
    epoch_schedulers: list[dict[str, Any]]
    checkpoint: CheckpointConf
    optimizer: OptimizerConf
    dataset: DatasetConfig

    @validator("epoch_schedulers", each_item=True)
    def assert_epoch_scheduler_exist(cls, sch):
        # TODO:
        return sch


class TrainingConfAccessor(TrainingConf):
    pass


# ################################
#     Validation configuration
# ################################
class ValidationConf(BaseModel):
    run_every_epoch: conint(ge=1)
    dataset: DatasetConfig


class ValidationConfAccessor(ValidationConf):
    pass


# ################################
#     Complete configuration
# ################################
class Conf(BaseModel):
    base: BaseConf
    model: ModelConf
    metrics: dict[str, dict[str, Any]] | None = Field(default_factory=dict)
    training: TrainingConf
    validator: ValidationConf

    def __init__(self, **data) -> None:
        super().__init__(**data)
        # TODO: validate the metric required by training.checkpoint.monitor is defined in metrics


class ConfAccessor(Conf):
    pass
