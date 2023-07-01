"""Module with stages utils."""
from enum import Enum


class Stage(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    PREDICT = "predict"

    def __str__(self) -> str:
        return self.value
