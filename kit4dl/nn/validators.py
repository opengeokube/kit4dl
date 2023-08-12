"""A module with resuable validators."""
import torch

from kit4dl.io import import_and_get_attr_from_fully_qualified_name
from kit4dl.mlkit_types import FullyQualifiedName


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
    assert torch.cuda.is_available(), "CUDA is not available"
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
