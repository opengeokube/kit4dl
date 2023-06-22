"""Docstring of the module containing my data module"""
from typing import Any

from torch.utils.data import Dataset

from mlkit import MLKitAbstractDataModule


class MyCustomDatamodule(MLKitAbstractDataModule):
    def prepare_traindataset(self, *args: Any, **kwargs: Any) -> Dataset:
        raise NotImplementedError

    def prepare_valdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        raise NotImplementedError

    def prepare_trainvaldataset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Dataset, Dataset]:
        # If you have single logic for train/val split,
        # implement this method and remove `prepare_traindataset`
        # and `prepare_valdataset`
        raise NotImplementedError

    def prepare_testdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        raise NotImplementedError

    def prepare_predictdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        raise NotImplementedError
