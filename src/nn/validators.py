"""A module with resuable validators"""
import importlib

import torch

from src.typing import FullyQualifiedName


def validate_cuda_device_exists(cuda_id: int | None = None) -> int | None:
    if cuda_id is None:
        return None
    assert (
        cuda_id < torch.cuda.device_count()
    ), f"cuda device with id `{cuda_id}` does not exist"
    return cuda_id


def validate_class_exists(path: FullyQualifiedName) -> FullyQualifiedName:
    if not "." in path:
        raise ValueError(
            "the provided path is not fully qualified name. remember to take"
            " into account package, like `package.subpackage.module.Class1`"
        )
    module_name, target = path.rsplit(".", maxsplit=1)
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ValueError(f"module: `{module_name}` could not be found")

    module = importlib.import_module(spec.name)
    if not hasattr(module, target):
        raise AttributeError(f"module `{module}` has not attribute `{target}`")
    return path
