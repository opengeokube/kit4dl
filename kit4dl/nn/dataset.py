"""A module with the Kit4DL abstract dataset definition."""

from __future__ import annotations

import os
from abc import ABC
from types import GeneratorType
from typing import Any, Callable, TYPE_CHECKING, Generator

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from kit4dl import context
from kit4dl.mixins import LoggerMixin

if TYPE_CHECKING:
    from kit4dl.nn.confmodels import DatasetConf


class Kit4DLAbstractDataModule(ABC, pl.LightningDataModule, LoggerMixin):
    """The class with the logic for dataset management.

    The class provides a user with a simple interface:
    `prepare_data` is a method for downloading and preprocessing datasets
    `prepare_traindatasets` is a method returning `torch.Dataset` for train
        data
    `prepare_valdatasets` is a method returning `torch.Dataset` for validation
        data
    `prepare_trainvaldatasets` is a method returning a tuple of two
        `torch.Dataset`s for train and validation data
        (if this method is provided, `prepare_traindatasets` and
        `prepare_valdatasets` shouldn't be implemented)
    `prepare_testdataset` is a method returning `torch.Dataset` for test data
    `prepare_predictdataset` is a method returning `torch.Dataset` for data
        for prediction
    """

    def __init__(self, conf: DatasetConf):
        super().__init__()
        self.conf = conf
        self.trainval_dataset_generator: (
            Generator[tuple[Dataset, Dataset], None, None] | None
        ) = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None
        self._split_suffix = os.environ.get(
            "KIT4DL_SPLIT_PREFIX", "(split={:02d})"
        )
        self._configure_logger()
        for extra_arg_key, extra_arg_value in self.conf.arguments.items():
            self.debug(
                "setting extra user-defined argument: %s:%s",
                extra_arg_key,
                extra_arg_value,
            )
            setattr(self, extra_arg_key, extra_arg_value)

    def _configure_logger(self) -> None:
        super().configure_logger(
            name="kit4dl.Dataset",
            level=context.LOG_LEVEL,
            logformat=context.LOG_FORMAT,
        )

    def prepare_data(self):
        """Prepare dataset for train/validation/test/predict splits.

        Examples
        --------
        ```python
        class MyDatamodule(Kit4DLAbstractDataModule):

            def prepare_data(self):
                # any logic you need to perform before creating splits
                download_dataset()
        ```
        """

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
        train_dataset : Dataset
            A training dataset

        Examples
        --------
        ```python
        ...
        def prepare_traindatasets(self, root_dir: str) -> Dataset:
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
        val_dataset : Dataset
            A validation dataset

        Examples
        --------
        ```python
        ...
        def prepare_valdatasets(self, root_dir: str) -> Dataset:
            val_dset = MyDataset(root_dir=root_dir)
            return val_dset
        ```
        """
        raise NotImplementedError

    def prepare_trainvaldatasets(
        self, *args: Any, **kwargs: Any
    ) -> Generator[tuple[Dataset, Dataset], None, None]:
        """Prepare dataset for training and validation.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the dataset
        **kwargs : Any
            List of named arguments required to setup the dataset

        Returns
        -------
        trainval_dataset_generators : tuple of two Datasets and a string
            Tuple consisting of train and validation dataset, and a suffix
            for the split

        Examples
        --------
        ```python
        ...
        def prepare_trainvaldatasets(self, root_dir: str) -> tuple[Dataset, Dataset]:
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
            self.trainval_dataset_generator = self.prepare_trainvaldatasets(
                **self.conf.trainval.arguments
            )
            if not isinstance(self.trainval_dataset_generator, GeneratorType):
                raise ValueError(
                    "The method `prepare_trainvaldatasets` should "
                    "be a generator (did you forget to use "
                    "`yield` keyword)"
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
            if not isinstance(self.train_dataset, Dataset):
                raise ValueError(
                    "The method `prepare_traindataset` should return a pytorch"
                    " Dataset. Did you forget to return a Dataset"
                )
            self.val_dataset = self.prepare_valdataset(
                **self.conf.validation.arguments
            )
            if not isinstance(self.val_dataset, Dataset):
                raise ValueError(
                    "The method `prepare_valdataset` should return a pytorch"
                    " Dataset. Did you forget to return a Dataset"
                )

    def _handle_test_stage(self) -> None:
        assert self.conf.test, (
            "configuration for test phase is not defined. did you forget to"
            " define `[dataset.test]` section in the configuration file?"
        )
        self.test_dataset = self.prepare_testdataset(
            **self.conf.test.arguments
        )
        if not isinstance(self.test_dataset, Dataset):
            raise ValueError(
                "The method `prepare_testdataset` should "
                "return a pytorch Dataset. Did you forget to return a Dataset"
            )

    def _handle_predict_stage(self) -> None:
        assert self.conf.predict, (
            "`test_config` is not defined. did you forget to define"
            " `[dataset.predict]` section in the configuration file?"
        )
        self.predict_dataset = self.prepare_predictdataset(
            **self.conf.predict.arguments
        )
        if not isinstance(self.predict_dataset, Dataset):
            raise ValueError(
                "The method `predict_dataset` should "
                "return a pytorch Dataset. Did you forget to return a Dataset"
            )

    def setup(self, stage: str) -> None:
        """Set up data depending on the stage.

        The method should not be overriden unless necessary.

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

    def get_collate_fn(self) -> Callable | None:
        """Get batch collate function."""
        return None

    def get_train_collate_fn(self) -> Callable | None:
        """Get train specific collate function."""
        return self.get_collate_fn()

    def get_val_collate_fn(self) -> Callable | None:
        """Get validation specific collate function."""
        return self.get_collate_fn()

    def get_test_collate_fn(self) -> Callable | None:
        """Get test specific collate function."""
        return self.get_collate_fn()

    def get_predict_collate_fn(self) -> Callable | None:
        """Get predict specific collate function."""
        return self.get_collate_fn()

    def trainval_dataloaders(
        self,
    ) -> Generator[tuple[DataLoader, DataLoader, str], None, None]:
        """Prepare loader for train and validation data."""
        if self.conf.trainval:
            assert self.trainval_dataset_generator is not None, (
                "did you forget to return `torch.utils.data.Dataset`instance"
                " from the `prepare_trainvaldatasets` method?"
            )
            assert self.conf.train is not None, (
                "[dataset.train.loader] section is is missing in the"
                " configuration file."
            )
            assert self.conf.validation is not None, (
                "[dataset.validation.loader] section is is missing in the"
                " configuration file."
            )
            assert (
                self.conf.train.loader is not None
                and self.conf.validation.loader is not None
            ), (
                "did you forget to define [dataset.train.loader] and"
                "[dataset.validation.loader] sections in the configuration "
                "file?"
            )
            for i, (tr_dataset, val_dataset) in enumerate(
                self.trainval_dataset_generator
            ):
                yield DataLoader(
                    tr_dataset,
                    **self.conf.train.loader,
                    collate_fn=self.get_train_collate_fn(),
                ), DataLoader(
                    val_dataset,
                    **self.conf.validation.loader,
                    collate_fn=self.get_val_collate_fn(),
                ), self._split_suffix.format(
                    i + 1
                )
        elif self.conf.train and self.conf.validation:
            yield self._train_dataloader(), self._val_dataloader(), ""

    def _train_dataloader(self) -> DataLoader:
        """Prepare loader for train data."""
        assert self.conf.train, (
            "train configuration is not defined. did you forget"
            " [dataset.train] section in the configuration file?"
        )
        assert self.train_dataset is not None, (
            "did you forget to return `torch.utils.data.Dataset` instance"
            " from the `prepare_traindataset` method?"
        )
        return DataLoader(
            self.train_dataset,
            **self.conf.train.loader,
            collate_fn=self.get_train_collate_fn(),
        )

    def _val_dataloader(self) -> DataLoader:
        """Prepare loader for validation data."""
        assert self.conf.validation, (
            "validation configuration is not defined. did you forget"
            " [dataset.validation] section in the configuration file?"
        )
        assert self.val_dataset is not None, (
            "did you forget to return `torch.utils.data.Dataset` instance"
            " from the `prepare_valdataset` method?"
        )
        return DataLoader(
            self.val_dataset,
            **self.conf.validation.loader,
            collate_fn=self.get_val_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        """Prepare loader for test data."""
        assert self.conf.test, (
            "test configuration is not defined. did you forget"
            " [dataset.test] section in the configuration file?"
        )
        assert self.test_dataset is not None, (
            "did you forget to return `torch.utils.data.Dataset` instance"
            " from the `prepare_testdataset` method?"
        )
        return DataLoader(
            self.test_dataset,
            **self.conf.test.loader,
            collate_fn=self.get_test_collate_fn(),
        )

    def predict_dataloader(self) -> DataLoader:
        """Prepare loader for prediction data."""
        assert self.conf.predict, (
            "validation configuration is not defined. did you forget"
            " [dataset.predict] section in the configuration file?"
        )
        assert self.predict_dataset is not None, (
            "did you forget to return `torch.utils.data.Dataset` instance"
            " from the `prepare_predictdataset` method?"
        )
        return DataLoader(
            self.predict_dataset,
            **self.conf.predict.loader,
            collate_fn=self.get_predict_collate_fn(),
        )

    def numpy_train_dataloader(self):
        """Prepare loader for train data for models accepting `numpy.ndarray`."""
        raise NotImplementedError

    def numpy_val_dataloader(self):
        """Prepare loader for val data for models accepting `numpy.ndarray`."""
        raise NotImplementedError

    def numpy_testdataloader(self):
        """Prepare loader for test data for models accepting `numpy.ndarray`."""
        raise NotImplementedError

    def numpy_predictdataloader(self):
        """Prepare loader for pred data for models accepting `numpy.ndarray`."""
        raise NotImplementedError
