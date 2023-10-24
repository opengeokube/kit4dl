"""A module with resuable validators."""

import warnings
from collections.abc import Collection
import torch
import torchmetrics as tm

import kit4dl.io as io_
from kit4dl import Kit4DLCallback
from kit4dl.io import import_and_get_attr_from_fully_qualified_name
from kit4dl.kit4dl_types import FullyQualifiedName

def split_extra_arguments(
    values: dict, fields: Collection, *, use_alias: bool = False
) -> tuple[dict, dict]:
    """Split arguments to field-related and auxiliary."""
    extra_args: dict = {}
    field_args: dict = {}
    if use_alias:
        for key, value in values.items():
            if key in fields:
                field_args.update({key: value})
                continue
            for f_info in fields:
                if key == f_info.alias:
                    field_args.update({key: value})
                    break
            else:
                extra_args.update({key: value})
    else:
        extra_args = {k: v for k, v in values.items() if k not in fields}
        field_args = {k: v for k, v in values.items() if k in fields}
    return (field_args, extra_args)

def build_arguments(values: dict, model_keys: Collection, use_alias: bool = False) -> dict:
    field_args, extra_args = split_extra_arguments(
        values, model_keys, use_alias=use_alias
    )
    if "arguments" not in field_args:
        field_args["arguments"] = extra_args
    return field_args

def validate_callback(conf: dict):
    """Assert callback exists."""
    assert "target" in conf, "`target` is not defined for some callbacks"
    target_class = io_.import_and_get_attr_from_fully_qualified_name(
        conf["target"]
    )
    assert issubclass(target_class, Kit4DLCallback), (
        "custom callbacks need to be subclasses of `kit4dl.Kit4DLCallback`"
        " class!"
    )
    return conf


def validate_metric(conf: dict):
    """Assert metric exists."""
    assert "target" in conf, "`target` is not defined for some metric"
    target_class = io_.import_and_get_attr_from_fully_qualified_name(
        conf["target"]
    )
    # TODO: issubclass for torchmetrics metric does not work
    # as __bases__ for metrics in `object`. Method issubclass
    # can be used for custom metrics
    _, attr_name = io_.split_target(conf["target"])
    assert issubclass(target_class, tm.Metric) or hasattr(
        tm, attr_name
    ), "custom metrics need to be subclasses of `torchmetrics.Metric` class!"
    return conf


def validate_lr_scheduler(conf: dict):
    """Assert the expected LR scheduler exists."""
    assert "target" in conf
    validate_class_exists(conf["target"])
    return conf


def validate_cuda_device_exists(cuda_id: int | None = None) -> int | None:
    """Assert the indicated CUDA device exists.

    Parameters
    ----------
    cuda_id : optional int
        ID of the CUDA device. `None` means CPU device

    Raises
    ------
    AssertionError
        if `cuda_id` is not `None` and CUDA is not available
        if `cuda_id` is not `None` and the requested device does not exist
    """
    if cuda_id is None:
        return None
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA device was specified but it is not available. CPU will be"
            " used!"
        )
        return None
    assert (
        cuda_id < torch.cuda.device_count()
    ), f"CUDA device with id `{cuda_id}` does not exist"
    return cuda_id


def validate_class_exists(
    path: FullyQualifiedName | str,
) -> FullyQualifiedName | str:
    """Validate the class defined by `path` exists.

    Parameters
    ----------
    path : str or FullyQualifiedName
        Path to the class to verify

    Returns
    -------
    path : str or FullyQualifiedName
        The path passed as an argument

    Raises
    ------
    ValueError
        if the path does not contain class separator `::`
    ModuleNotFoundError
        if module was not found
    """
    _ = import_and_get_attr_from_fully_qualified_name(path)
    return path
