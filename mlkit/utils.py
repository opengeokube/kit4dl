from typing import Callable, Hashable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def replace_item_recursively(entries: dict, key: Hashable, replace: Callable):
    for k, v in entries.items():
        if isinstance(v, dict):
            entries[k] = replace_item_recursively(v, key, replace)
    if key in entries:
        entries[key] = replace(entries[key])
    return entries
