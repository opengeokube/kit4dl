"""A module with the MLKit abstract dataset definition"""
from abc import ABC
from typing import Any

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from mlkit.nn.confmodels import DatasetConf


class MLKitAbstractDataModule(ABC, pl.LightningDataModule):
    """The class with the logic for dataset management.
    The class provides a user with a simple interface:

    `prepare_data` is a method for downloading and preprocessing datasets
    `prepare_traindataset` is a method returning `torch.Dataset` for train
        data
    `prepare_valdataset` is a method returning `torch.Dataset` for validation
        data
    `prepare_trainvaldataset` is a method returning a tuple of two
        `torch.Dataset`s for train and validation data
        (if this method is provided, `prepare_traindataset` and
        `prepare_valdataset` shouldn't be implemented)
    `prepare_testdataset` is a method returning `torch.Dataset` for test data
    `prepare_predictdataset` is a method returning `torch.Dataset` for data
        for prediction
    """

    def __init__(self, conf: DatasetConf):
        super().__init__()
        self.conf = conf
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None

    def prepare_data(self):
        pass

    def prepare_traindataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare dataset for training.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the dataset
        **kwargs : Any
            List of named arguments required to setup the dataset

        Returns
        -------
        train_datasets : Dataset
            A training dataset

        Examples
        --------
        ```python
        ...
        def prepare_traindataset(self, root_dir: str) -> Dataset:
            train_dset = MyDataset(root_dir=root_dir)
            return train_dset
        ```
        """
        raise NotImplementedError

    def prepare_valdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare dataset for validation.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the dataset
        **kwargs : Any
            List of named arguments required to setup the dataset

        Returns
        -------
        val_datasets : Dataset
            A validation dataset

        Examples
        --------
        ```python
        ...
        def prepare_valdataset(self, root_dir: str) -> Dataset:
            val_dset = MyDataset(root_dir=root_dir)
            return val_dset
        ```
        """
        raise NotImplementedError

    def prepare_trainvaldataset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Dataset, Dataset]:
        """Prepare dataset for training and validation.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the dataset
        **kwargs : Any
            List of named arguments required to setup the dataset

        Returns
        -------
        trainval_datasets : tuple of Dataset
            Tuple consisting of train and validation dataset

        Examples
        --------
        ```python
        ...
        def prepare_trainvaldataset(self, root_dir: str) -> tuple[Dataset, Dataset]:
            dset = MyDataset(root_dir=root_dir)
            train_dset, val_dset = random_split(dset, [500, 50])
            return train_dset, val_dset
        ```
        """
        raise NotImplementedError

    def prepare_testdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare dataset for testing.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the dataset
        **kwargs : Any
            List of named arguments required to setup the dataset

        Returns
        -------
        test_datasets : Dataset
            A test dataset

        Examples
        --------
        ```python
        ...
        def prepare_testdataset(self, root_dir: str) -> Dataset:
            return MyDataset(root_dir=root_dir)
        ```
        """
        raise NotImplementedError

    def prepare_predictdataset(self, *args: Any, **kwargs: Any) -> Dataset:
        """Prepare dataset for predicting.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the dataset
        **kwargs : Any
            List of named arguments required to setup the dataset

        Returns
        -------
        pred_datasets : Dataset
            A prediction dataset

        Examples
        --------
        ```python
        ...
        def prepare_predictdataset(self, root_dir: str) -> Dataset:
            return MyDataset(root_dir=root_dir)
        ```
        """
        raise NotImplementedError

    def _handle_fit_stage(self) -> None:
        if self.conf.trainval:
            assert self.conf.trainval, (
                "configuration for train-validation phases is not defined. did"
                " you forget to define `[dataset.trainval]` section in the"
                " configuration file?"
            )
            self.train_dataset, self.val_dataset = (
                self.prepare_trainvaldataset(**self.conf.trainval.arguments)
            )
        else:
            assert self.conf.train, (
                "configuration for train phase is not defined. did you forget"
                " to define `[dataset.train]` section in the configuration"
                " file?"
            )
            assert self.conf.validation, (
                "configuration for validation phase is not defined. did you"
                " forget to define `[dataset.validation]` section in the"
                " configuration file?"
            )
            self.train_dataset = self.prepare_traindataset(
                **self.conf.train.arguments
            )
            self.val_dataset = self.prepare_valdataset(
                **self.conf.validation.arguments
            )

    def _handle_test_stage(self) -> None:
        assert self.conf.test, (
            "configuration for test phase is not defined. did you forget to"
            " define `[dataset.test]` section in the configuration file?"
        )
        self.val_dataset = self.prepare_testdataset(**self.conf.test.arguments)

    def _handle_predict_stage(self) -> None:
        assert self.conf.predict, (
            "`test_config` is not defined. did you forget to define"
            " `[dataset.predict]` section in the configuration file?"
        )
        self.predict_dataset = self.prepare_predictdataset(
            **self.conf.predict.arguments
        )

    def setup(self, stage: str) -> None:
        """Setup data depending on the stage. The method should not be
        overriden unless necessary.

        Parameters
        ----------
        stage : str
            The stage of the pipeline. One out of `['fit', 'test', 'predict']`
        """
        match stage:
            case "fit":
                self._handle_fit_stage()
            case "test":
                self._handle_test_stage()
            case "predict":
                self._handle_predict_stage()

    def train_dataloader(self) -> DataLoader:
        """Prepare loader for train data"""
        assert self.train_dataset is not None, (
            "`train_dataset` must be set before calling `train_dataloader`"
            " method!"
        )
        return DataLoader(self.train_dataset, **self.conf.train.loader)

    def val_dataloader(self) -> DataLoader:
        """Prepare loader for validation data"""
        assert self.conf.validation, (
            "validation configuration is not defined. did you forget"
            " [dataset.validation] section in the configuration file?"
        )
        assert self.val_dataset
        return DataLoader(self.val_dataset, **self.conf.validation.loader)

    def test_dataloader(self) -> DataLoader:
        """Prepare loader for test data"""
        assert self.conf.test, (
            "validation configuration is not defined. did you forget"
            " [dataset.test] section in the configuration file?"
        )
        assert self.test_dataset
        return DataLoader(self.test_dataset, **self.conf.test.loader)

    def predict_dataloader(self) -> DataLoader:
        """Prepare loader for prediction data"""
        assert self.conf.predict, (
            "validation configuration is not defined. did you forget"
            " [dataset.predict] section in the configuration file?"
        )
        assert self.predict_dataset
        return DataLoader(self.predict_dataset, **self.conf.predict.loader)

    def numpy_train_dataloader(self):
        """Prepare loader for train data for models accepting `numpy.ndarray`"""
        raise NotImplementedError

    def numpy_val_dataloader(self):
        """Prepare loader for val data for models accepting `numpy.ndarray`"""
        raise NotImplementedError

    def numpy_testdataloader(self):
        """Prepare loader for test data for models accepting `numpy.ndarray`"""
        raise NotImplementedError

    def numpy_predictdataloader(self):
        """Prepare loader for predict data for models accepting `numpy.ndarray`"""
        raise NotImplementedError
