"""Docstring of the module containing my data module."""

from typing import Any, Generator

from torch.utils.data import Dataset

from kit4dl.nn.dataset import Kit4DLAbstractDataModule


class MyCustomDatamodule(Kit4DLAbstractDataModule):
    """My datamodule docstring."""

    def prepare_traindatasets(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare train dataset."""
        raise NotImplementedError

    def prepare_valdatasets(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare validation dataset."""
        raise NotImplementedError

    def prepare_trainvaldatasets(
        self, *args: Any, **kwargs: Any
    ) -> Generator[tuple[Dataset, Dataset], None, None]:
        """Prepare train and validation dataset."""
        # NOTE: If you have single logic for train/val split,
        # implement this method and remove `prepare_traindatasets`
        # and `prepare_valdatasets`
        raise NotImplementedError

    def prepare_testdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare test dataset."""
        raise NotImplementedError

    def prepare_predictdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare predict dataset."""
        raise NotImplementedError
