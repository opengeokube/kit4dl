"""A module with resuable validators"""
import importlib
import os

import torch

from mlkit.io import import_and_get_attr_from_fully_qualified_name
from mlkit.types import FullyQualifiedName


def validate_cuda_device_exists(cuda_id: int | None = None) -> int | None:
    if cuda_id is None:
        return None
    assert torch.cuda.is_available(), "CUDA is not available"
    assert (
        cuda_id < torch.cuda.device_count()
    ), f"CUDA device with id `{cuda_id}` does not exist"
    return cuda_id


def validate_class_exists(
    path: FullyQualifiedName | str | os.PathLike,
) -> FullyQualifiedName | str | os.PathLike:
    _ = import_and_get_attr_from_fully_qualified_name(path)
    return path
