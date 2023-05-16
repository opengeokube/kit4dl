"""Docstring of the module containing my data module"""
from torch.utils.data import Dataset

from mlkit import MLKitAbstractDataset


class MyCustomDatamodule(MLKitAbstractDataset):
    def prepare_traindataset(self, **kwargs) -> Dataset:
        raise NotImplementedError

    def prepare_valdataset(self, **kwargs) -> Dataset:
        raise NotImplementedError

    def prepare_trainvaldataset(self, **kwargs) -> tuple[Dataset, Dataset]:
        # If you have single logic for train/val split,
        # implement this method and remove `prepare_traindataset`
        # and `prepare_valdataset`
        raise NotImplementedError

    def prepare_testdataset(self, **kwargs) -> Dataset:
        raise NotImplementedError

    def prepare_predictdataset(self, **kwargs) -> Dataset:
        raise NotImplementedError
