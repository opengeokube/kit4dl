"""A module with the MLKit abstract dataset definition"""
from abc import ABC, abstractmethod

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from mlkit.nn.confmodels import DatasetConfig


class MLKitAbstractDataset(ABC, pl.LightningDataModule):
    def __init__(
        self,
        train_config: DatasetConfig,
        val_config: DatasetConfig | None = None,
        test_config: DatasetConfig | None = None,
        predict_config: DatasetConfig | None = None,
    ):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config
        self.predict_config = predict_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        pass

    @abstractmethod
    def prepare_traindataset(self, **kwargs) -> Dataset:
        """Prepare dataset for training.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

        Examples
        --------
        ```python
        ...
        def prepare_traindataset(self, root_dir: str) -> Dataset:
            return MyDataset(root_dir=root_dir)
        ```
        """
        raise NotImplementedError

    def prepare_valdataset(self, **kwargs) -> Dataset:
        """Prepare dataset for validation.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

        Examples
        --------
        ```python
        ...
        def prepare_vadataset(self, root_dir: str) -> Dataset:
            return MyDataset(root_dir=root_dir)
        ```
        """
        raise NotImplementedError

    def prepare_testdataset(self, **kwargs) -> Dataset:
        """Prepare dataset for testing.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

        Examples
        --------
        ```python
        ...
        def prepare_testdataset(self, root_dir: str) -> Dataset:
            return MyDataset(root_dir=root_dir)
        ```
        """
        raise NotImplementedError

    def prepare_predictdataset(self, **kwargs) -> Dataset:
        """Prepare dataset for predicting.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

        Examples
        --------
        ```python
        ...
        def prepare_predictdataset(self, root_dir: str) -> Dataset:
            return MyDataset(root_dir=root_dir)
        ```
        """
        raise NotImplementedError

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.train_dataset = self.prepare_traindataset(
                    **self.train_config.dataset_kwargs
                )
                assert not isinstance(self.train_dataset, Dataset), (
                    "method `prepare_traindataset` must return object of class"
                    " `torch.utils.data.Dataset` or its subclass"
                )
                if self.val_config is not None:
                    self.val_dataset = self.prepare_valdataset(
                        **self.val_config.dataset_kwargs
                    )
                    assert not isinstance(self.val_dataset, Dataset), (
                        "method `prepare_valdataset` must return object of"
                        " class `torch.utils.data.Dataset` or its subclass"
                    )
            case "test":
                assert self.test_config is not None, (
                    "`test_config` is not defined. did you forget to define"
                    " `[test]` section in the configuration file?"
                )
                self.val_dataset = self.prepare_testdataset(
                    **self.test_config.dataset_kwargs
                )
                assert not isinstance(self.val_dataset, Dataset), (
                    "method `prepare_testdataset` must return object of class"
                    " `torch.utils.data.Dataset` or its subclass"
                )
            case "predict":
                assert self.test_config is not None, (
                    "`test_config` is not defined. did you forget to define"
                    " `[predict]` section in the configuration file?"
                )
                self.predict_dataset = self.prepare_predictdataset(
                    **self.predict_config.dataset_kwargs
                )
                assert not isinstance(self.predict_dataset, Dataset), (
                    "method `prepare_predictdataset` must return object of"
                    " class `torch.utils.data.Dataset` or its subclass"
                )

    def train_dataloader(self):
        assert self.train_dataset is not None, (
            "`train_dataset` must be set before calling `train_dataloader`"
            " method!"
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=self.train_config.shuffle,
            num_workers=self.train_config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            shuffle=self.val_config.shuffle,
            num_workers=self.val_config.num_workers,
        )

    def test_dataloader(self):
        assert (
            self.predict_config is not None
        ), "configuration file was not defined for TESTING"
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            shuffle=self.test_config.shuffle,
            num_workers=self.test_config.num_workers,
        )

    def predict_dataloader(self):
        assert (
            self.predict_config is not None
        ), "configuration file was not defined for PREDICTION"
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_config.batch_size,
            shuffle=self.predict_config.shuffle,
            num_workers=self.predict_config.num_workers,
        )

    def numpy_train_dataloader(self):
        pass

    def numpy_val_dataloader(self):
        pass

    def numpy_testdataloader(self):
        pass
