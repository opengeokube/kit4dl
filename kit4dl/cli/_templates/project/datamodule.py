"""Docstring of the module containing my data module."""
from typing import Any

from torch.utils.data import Dataset

from kit4dl import Kit4DLAbstractDataModule


class MyCustomDatamodule(Kit4DLAbstractDataModule):
    """My datamodule docstring."""

    def prepare_traindataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare train dataset."""
        raise NotImplementedError

    def prepare_valdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare validation dataset."""
        raise NotImplementedError

    def prepare_trainvaldataset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Dataset, Dataset]:
        """Prepare train and validation dataset."""
        # NOTE: If you have single logic for train/val split,
        # implement this method and remove `prepare_traindataset`
        # and `prepare_valdataset`
        raise NotImplementedError

    def prepare_testdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare test dataset."""
        raise NotImplementedError

    def prepare_predictdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare predict dataset."""
        raise NotImplementedError
