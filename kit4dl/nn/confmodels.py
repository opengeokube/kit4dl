"""A module with configuration classes."""
import os
import warnings
from functools import partial as func_partial
from inspect import signature
from typing import Any, Callable, Literal

import lightning.pytorch.loggers as pl_logs
import torch
import torchmetrics as tm
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from typing_extensions import Annotated

import kit4dl.io as io_
from kit4dl import Kit4DLCallback
from kit4dl import utils as ut
from kit4dl.kit4dl_types import FullyQualifiedName
from kit4dl.nn.validators import (
    validate_callback,
    validate_class_exists,
    validate_cuda_device_exists,
    validate_lr_scheduler,
    validate_metric,
)

Target = Annotated[
    FullyQualifiedName | str, AfterValidator(validate_class_exists)
]
CudaDevice = Annotated[int | None, AfterValidator(validate_cuda_device_exists)]
SchedulerDict = Annotated[
    dict[str, Any], AfterValidator(validate_lr_scheduler)
]
MetricDict = Annotated[dict[str, Any], AfterValidator(validate_metric)]
CallbackDict = Annotated[dict[str, Any], AfterValidator(validate_callback)]


def split_extra_arguments(
    values: dict, fields: dict[str, FieldInfo], *, consider_alias: bool = False
) -> tuple[dict, dict]:
    """Split arguments to field-related and auxiliary."""
    extra_args: dict = {}
    field_args: dict = {}
    if consider_alias:
        for key, value in values.items():
            if key in fields:
                field_args.update({key: value})
                continue
            for f_info in fields.values():
                if key == f_info.alias:
                    field_args.update({key: value})
                    break
            else:
                extra_args.update({key: value})
    else:
        extra_args = {k: v for k, v in values.items() if k not in fields}
        field_args = {k: v for k, v in values.items() if k in fields}
    return (field_args, extra_args)


def create_obj_from_conf(obj_conf: dict, partial: bool = False) -> Any:
    """Initialize object or prepared `partial` object based on configuration."""
    obj_conf_copy = obj_conf.copy()
    class_ = io_.import_and_get_attr_from_fully_qualified_name(
        obj_conf_copy.pop("target")
    )
    if partial:
        return func_partial(
            class_,
            **obj_conf_copy,
        )
    return class_(**obj_conf_copy)


# ################################
#           ABSTRACT
# ################################
class _AbstractClassWithArgumentsConf(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    target: Target
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        field_args, extra_args = split_extra_arguments(
            values, cls.model_fields, consider_alias=False
        )
        field_args["arguments"] = extra_args
        return field_args


# ################################
#  Basic experiment configuration
# ################################
class BaseConf(BaseModel):
    """Base configuration model for the experiment."""

    seed: int = Field(default=0, ge=0)
    cuda_id: CudaDevice = None
    experiment_name: str

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

    @field_validator("target")
    def _check_if_target_has_expected_parent_class(cls, value):
        from kit4dl.nn.base import (  # pylint: disable=import-outside-toplevel
            Kit4DLAbstractModule,
        )

        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, Kit4DLAbstractModule), (
            f"target class must be a subclass of `{Kit4DLAbstractModule}`"
            " class!"
        )
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

    @field_validator("target")
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
        return func_partial(
            target_class,
            lr=self.lr,
            **self.arguments,  # pylint: disable=not-a-mapping
        )


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

    @field_validator("path")
    def _warn_on_existing_path(cls, path: str):
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            warnings.warn(
                f"directory {path} exists and is not empty", ResourceWarning
            )
        return path

    @field_validator("monitor")
    def _assert_required_monitor_keys_defined(cls, monitor: dict):
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

    target: Target
    weight: list[float] | None = None

    @field_validator("target")
    def _check_if_target_has_expected_parent_class(cls, value):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, torch.nn.Module), (
            "target class of the criterion must be a subclass of"
            f" `{torch.nn.Module}` class!"
        )
        return value

    @model_validator(mode="after")
    def _match_weight(cls, values):
        if values.weight:
            criterion_class = (
                io_.import_and_get_attr_from_fully_qualified_name(
                    values.target
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
    epoch_schedulers: list[SchedulerDict] = Field(default_factory=list)
    checkpoint: CheckpointConf | None = None
    callbacks: list[CallbackDict] = Field(default_factory=list)
    optimizer: OptimizerConf
    criterion: CriterionConf
    arguments: dict[str, Any]

    @model_validator(mode="before")
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        field_args, extra_args = split_extra_arguments(
            values, cls.model_fields, consider_alias=False
        )
        field_args["arguments"] = extra_args
        return field_args

    @property
    def preconfigured_callbacks(self) -> list[Kit4DLCallback]:
        """Get list of all preconfigured callbacks."""
        callbacks: list[Kit4DLCallback] = []
        for clb in self.callbacks:  # pylint: disable=not-an-iterable
            callbacks.append(create_obj_from_conf(clb, partial=False))
        return callbacks

    @property
    def preconfigured_schedulers_classes(
        self,
    ) -> list[Callable]:
        """Get a list of preconfigured schedulers."""
        schedulers: list[Callable] = []

        for sch in self.epoch_schedulers:  # pylint: disable=not-an-iterable
            schedulers.append(create_obj_from_conf(sch, partial=True))
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

    @model_validator(mode="before")
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        field_args, extra_args = split_extra_arguments(
            values, cls.model_fields, consider_alias=False
        )
        field_args["arguments"] = extra_args
        return field_args


# ################################
#     Dataset configuration
# ################################
class DatasetConf(BaseModel):
    """Dataset configuration class."""

    target: Target
    train: SplitDatasetConf | None = None
    validation: SplitDatasetConf | None = None
    trainval: SplitDatasetConf | None = None
    test: SplitDatasetConf | None = None
    predict: SplitDatasetConf | None = None
    arguments: dict[str, Any]

    @model_validator(mode="before")
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        field_args, extra_args = split_extra_arguments(
            values, cls.model_fields, consider_alias=False
        )
        field_args["arguments"] = extra_args
        return field_args

    @field_validator("target")
    def _check_if_target_has_expected_parent_class(cls, value: str):
        from kit4dl.dataset import (  # pylint: disable=import-outside-toplevel
            Kit4DLAbstractDataModule,
        )

        target_class = io_.import_and_get_attr_from_fully_qualified_name(value)
        assert issubclass(target_class, Kit4DLAbstractDataModule), (
            "target class of the dataset module must be a subclass of"
            f" `{Kit4DLAbstractDataModule}` class!"
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


class LoggingConf(BaseModel):
    """Logging configuration class."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    type_: Literal[
        "comet", "csv", "mlflow", "neptune", "tensorboard", "wandb"
    ] = Field("csv", alias="type")
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"] | None = (
        "INFO"
    )
    format_: str | None = Field("%(asctime)s - %(message)s", alias="format")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def _build_model_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
        field_args, extra_args = split_extra_arguments(
            values, cls.model_fields, consider_alias=True
        )
        field_args["arguments"] = extra_args
        return field_args

    @field_validator("level", mode="before")
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
        if issubclass(ltype, pl_logs.CSVLogger):
            self.arguments.setdefault("save_dir", "./csv_logs")

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
    metrics: dict[str, MetricDict] = Field(default_factory=dict)
    training: TrainingConf
    validation: ValidationConf
    dataset: DatasetConf

    def __init__(self, root_dir: str | None = None, **kwargs):
        if root_dir:
            kwargs = Conf.override_with_abs_target(root_dir, kwargs)
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def _update_experiment_name_if_undefined(cls, value):
        if not value.base:
            return value
        if value.logging:
            value.logging.maybe_update_experiment_name(
                value.base.experiment_name
            )
        return value

    @model_validator(mode="after")
    def _check_metric_in_checkpoint_is_defined(cls, value):
        if not value.training.checkpoint:
            return value
        monitored_metric_name = value.training.checkpoint.monitor["metric"]
        assert monitored_metric_name in value.metrics.keys(), (
            f"metric `{monitored_metric_name}` is not defined. did you forget"
            f" to define `{monitored_metric_name}` in [metrics]?"
        )
        return value

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
                )(**metric_args_copy).to(self.base.device)
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
        replace_logic = func_partial(
            io_.maybe_get_abs_target, root_dir=root_dir
        )
        entries = ut.replace_item_recursively(
            entries, key="target", replace=replace_logic
        )
        return entries
