"""A module with configuration classes."""
import os
import warnings
from abc import ABC
from functools import partial
from inspect import signature
from typing import Any, Callable, Literal

import lightning.pytorch.loggers as pl_logs
import torch
import torchmetrics as tm
from pydantic import BaseModel, Field, root_validator, validator

import mlkit.io as io_
from mlkit import utils as ut
from mlkit.mlkit_types import FullyQualifiedName
from mlkit.nn.validators import (
    validate_class_exists,
    validate_cuda_device_exists,
)


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
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
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
    """Base configuration model for the experiment."""

    seed: int = Field(default=0, ge=0)
    cuda_id: int | None = None
    experiment_name: str

    _assert_cuda_device = validator("cuda_id", allow_reuse=True)(
        validate_cuda_device_exists
    )

    @property
    def device(self) -> torch.device:
        """Get PyTorch device based on the provided configuration."""
        if self.cuda_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.cuda_id}")

    @property
    def accelerator_device_and_id(self) -> tuple[str, list[int] | str]:
        """Get an accelerator name and its ID.

        Get accelerator name: cpu or gpu and ID of the device.
        For CPU returns instead of ID "auto" is returned.
        """
        if self.cuda_id is None:
            return ("cpu", "auto")
        return ("gpu", [self.cuda_id])


# ################################
#  Neural network model configuration
# ################################
class ModelConf(_AbstractClassWithArgumentsConf):
    """Model configuration class."""

    @validator("target")
    def _check_if_target_has_expected_parent_class(cls, value):
        from mlkit.nn.base import (  # pylint: disable=import-outside-toplevel
            MLKitAbstractModule,
        )

        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(
            target_class, MLKitAbstractModule
        ), f"target class must be a subclass of `{MLKitAbstractModule}` class!"
        return value

    @property
    def model_class(self) -> type:
        """Get Python class of the model defined in the configuration."""
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        return target_class


# ################################
#       Optimizer configuration
# ################################
class OptimizerConf(_AbstractClassWithArgumentsConf):
    """Optimizer configuration class."""

    lr: float = Field(gt=0)

    @validator("target")
    def _check_if_target_has_expected_parent_class(cls, value):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, torch.optim.Optimizer), (
            "target class of the optimizer must be a subclass of"
            f" `{torch.optim.Optimizer}` class!"
        )
        return value

    @property
    def optimizer(self) -> Callable[..., torch.optim.Optimizer]:
        """Get `partial` object of the preconfigured optimizer."""
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        return partial(target_class, lr=self.lr, **self.arguments)


# ################################
#       Checkpoint configuration
# ################################
class CheckpointConf(BaseModel):
    """Checkpoint configuration class."""

    path: str
    monitor: dict[str, str]
    filename: str
    mode: Literal["min", "max"] = "max"
    save_top_k: int = Field(1, ge=1)
    save_weights_only: bool = True
    every_n_epochs: int = Field(1, ge=1)
    save_on_train_epoch_end: bool = False

    @validator("path")
    def _warn_on_existing_path(cls, path):
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            warnings.warn(
                f"directory {path} exists and is not empty", ResourceWarning
            )
        return path

    @validator("monitor")
    def _assert_required_monitor_keys_defined(cls, monitor):
        assert "metric" in monitor, "`metric` key is missing"
        assert (
            "stage" in monitor
        ), "`stage` key is missing. define `stage='train'` or `stage='val'`"
        assert monitor["stage"] in {"train", "val"}
        return monitor

    @property
    def monitor_metric_name(self) -> str:
        """Returns the name of the metric to be monitored."""
        return "_".join(
            [self.monitor["stage"].lower(), self.monitor["metric"].lower()]
        )


# ################################
#     Criterion configuration
# ################################
class CriterionConf(BaseModel):
    """Criterion configuration class."""

    target: FullyQualifiedName
    weight: list[float] | None = None

    _validate_class_exists = validator("target", allow_reuse=True)(
        validate_class_exists
    )

    @validator("target")
    def _check_if_target_has_expected_parent_class(cls, value):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, torch.nn.Module), (
            "target class of the criterion must be a subclass of"
            f" `{torch.nn.Module}` class!"
        )
        return value

    @root_validator(skip_on_failure=True)
    def _match_weight(cls, values):
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
        """Get the torch.nn.Module with the criterion function."""
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
    """Training procedure configuration class."""

    epochs: int = Field(gt=0)
    epoch_schedulers: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint: CheckpointConf | None = None
    optimizer: OptimizerConf
    criterion: CriterionConf
    arguments: dict[str, Any]

    @root_validator(pre=True)
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "arguments" in values:
            return values
        arguments = {
            k: v for k, v in values.items() if k not in cls.__fields__
        }
        values = {k: v for k, v in values.items() if k in cls.__fields__}
        values["arguments"] = arguments
        return values

    @validator("epoch_schedulers", each_item=True)
    def _assert_epoch_scheduler_exist(cls, sch):
        assert "target" in sch
        validate_class_exists(sch["target"])
        return sch

    @property
    def preconfigured_schedulers_classes(
        self,
    ) -> list[Callable]:
        """Get a list of preconfigured schedulers."""
        schedulers: list[Callable] = []

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
    """Validation procedure configuration class."""

    run_every_epoch: int = Field(gt=0)


# ################################
#     Split dataset configuration
# ################################
class SplitDatasetConf(BaseModel):
    """Configuration class with dataset split arguments."""

    loader: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, Any]

    @root_validator(pre=True)
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
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
    """Dataset configuration class."""

    target: FullyQualifiedName | str
    train: SplitDatasetConf | None = None
    validation: SplitDatasetConf | None = None
    trainval: SplitDatasetConf | None = None
    test: SplitDatasetConf | None = None
    predict: SplitDatasetConf | None = None
    arguments: dict[str, Any]

    @root_validator(pre=True)
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "arguments" in values:
            return values
        arguments = {
            k: v for k, v in values.items() if k not in cls.__fields__
        }
        values = {k: v for k, v in values.items() if k in cls.__fields__}
        values["arguments"] = arguments
        return values

    @validator("target")
    def _check_if_target_has_expected_parent_class(cls, value):
        from mlkit.dataset import (  # pylint: disable=import-outside-toplevel
            MLKitAbstractDataModule,
        )

        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, MLKitAbstractDataModule), (
            "target class of the dataset module must be a subclass of"
            f" `{MLKitAbstractDataModule}` class!"
        )
        return value

    @property
    def datamodule_class(self) -> type:
        """Get Python class of the data module."""
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            self.target
        )
        return target_class


# ################################
#     Logging configuration
# ################################
_LOGGERS_NICKNAMES: dict[str, str] = {
    "comet": "CometLogger",
    "csv": "CSVLogger",
    "mlflow": "MLFlowLogger",
    "neptune": "NeptuneLogger",
    "tensorboard": "TensorBoardLogger",
    "wandb": "WandbLogger",
}


class LoggingConf(
    BaseModel, allow_population_by_field_name=True, extra="allow"
):
    """Logging configuration class."""

    type_: Literal[
        "comet", "csv", "mlflow", "neptune", "tensorboard", "wandb"
    ] = Field("csv", alias="type")
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"] | None = (
        "INFO"
    )
    format_: str | None = Field("%(asctime)s - %(message)s", alias="format")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @root_validator(pre=False)
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        arguments = {
            k: v for k, v in values.items() if k not in cls.__fields__
        }
        values = {k: v for k, v in values.items() if k in cls.__fields__}
        values["arguments"] = arguments
        return values

    @validator("level", pre=True)
    def _match_log_level(cls, item):
        return item.upper()

    def maybe_update_experiment_name(self, experiment_name: str) -> None:
        """Update experiment name for the chosen metric logger."""
        ltype = self.metric_logger_type
        if issubclass(ltype, (pl_logs.CometLogger, pl_logs.MLFlowLogger)):
            self.arguments.setdefault("experiment_name", experiment_name)
        elif issubclass(
            ltype,
            (
                pl_logs.CSVLogger,
                pl_logs.NeptuneLogger,
                pl_logs.TensorBoardLogger,
                pl_logs.WandbLogger,
            ),
        ):
            self.arguments.setdefault("name", experiment_name)
        else:
            raise TypeError(
                f"logger of type `{self.metric_logger_type}` is undefined!"
            )

    @property
    def metric_logger_type(self) -> type:
        """Get type of the selected logger."""
        return getattr(pl_logs, _LOGGERS_NICKNAMES[self.type_])


# ################################
#     Complete configuration
# ################################
class Conf(BaseModel):
    """Conf class being the reflection of the configuration TOML file."""

    base: BaseConf
    logging: LoggingConf = Field(default_factory=LoggingConf)  # type: ignore[arg-type]
    model: ModelConf
    metrics: dict[str, dict[str, Any]] | None = Field(default_factory=dict)
    training: TrainingConf
    validation: ValidationConf
    dataset: DatasetConf

    def __init__(self, root_dir: str | None = None, **kwargs):
        if root_dir:
            kwargs = Conf.override_with_abs_target(root_dir, kwargs)
        super().__init__(**kwargs)

    @validator("logging")
    def _update_experiment_name_if_undefined(cls, value: LoggingConf, values):
        if "base" not in values:
            return value
        value.maybe_update_experiment_name(values["base"].experiment_name)
        return value

    @validator("metrics")
    def _validate_metrics_class_exist(cls, values):
        if not values:
            return None
        for metric_name, metric_dict in values.items():
            assert (
                "target" in metric_dict
            ), f"`target` is not defined for the metric `{metric_name}`"
            target_class = io_.import_and_get_attr_from_fully_qualified_name(
                metric_dict["target"]
            )
            # TODO: issubclass for torchmetrics metric does not work
            # as __bases__ for metrics in `object`. Method issubclass
            # can be used for custom metrics
            _, attr_name = io_.split_target(metric_dict["target"])
            assert issubclass(target_class, tm.Metric) or hasattr(
                tm, attr_name
            ), (
                "custom metrics need to be subclasses of `torchmetrics.Metric`"
                " class!"
            )
        return values

    @root_validator(skip_on_failure=True)
    def _check_metric_in_checkpoint_is_defined(cls, values):
        monitored_metric_name = values["training"].checkpoint.monitor["metric"]
        assert monitored_metric_name in values["metrics"].keys(), (
            f"metric `{monitored_metric_name}` is not defined. did you forget"
            f" to define `{monitored_metric_name}` in [metrics]?"
        )
        return values

    @property
    def metrics_obj(self) -> dict[str, tm.Metric]:
        """Get the dictionary of the metric name and torchmetrics.Metric."""
        if not self.metrics:
            raise ValueError("metrics are not defined!")
        metric_obj: dict = {}
        for metric_name, metric_args in self.metrics.items():
            metric_args_copy = metric_args.copy()
            metric_obj[metric_name.lower()] = (
                io_.import_and_get_attr_from_fully_qualified_name(
                    metric_args_copy.pop("target")
                )(**metric_args_copy)
            )
        return metric_obj

    @classmethod
    def override_with_abs_target(cls, root_dir: str, entries: dict) -> dict:
        """Replace in-place `target` valuesof the `entries`.

        Parameters
        ----------
        root_dir : str
            Root dir to consider as an initial part of the absolute path
        entries : dict
            Dictionary of values to replace `target` key

        Returns
        -------
        replaced : dict
            In-place modified `entries` dictionary
        """
        replace_logic = partial(io_.maybe_get_abs_target, root_dir=root_dir)
        entries = ut.replace_item_recursively(
            entries, key="target", replace=replace_logic
        )
        return entries
