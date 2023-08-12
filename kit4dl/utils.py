"""Module with utilities functions and structures."""
import os
import random
from typing import Callable, Hashable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for random numbers for NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Seed value
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def replace_item_recursively(
    entries: dict, key: Hashable, replace: Callable
) -> dict:
    """Replace in-place value for `key` recursively.

    Replaces recursively values for `key` by the result of the `replace`
    callable. If value of `key` is a dictionary, the method iterates over it.

    Parameters
    ----------
    entries : dict
        A dictionary or dict of dict whose values for the `key`
        are to be replaced
    key : hashable
        A key, whose values will be replaced
    replace : callable
        A function taking as an argument the value of dictionary associated
        with `key`

    Returns
    -------
    res : dict
        A modified dictionary itself

    Examples
    --------
    ```python
    >>> my_dict = {"a": 1, "b": {"a": -1}}
    >>> replace_item_recursively(my_dict, "a", lambda val: val*10)
    {'a': 10, 'b': {'a': -10}}
    ```
    """
    for it_key, value in entries.items():
        if isinstance(value, dict):
            replace_item_recursively(value, key, replace)
        if key == it_key:
            entries[key] = replace(entries[key])
    return entries
