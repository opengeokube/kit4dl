"""A module with the MLKit abstract dataset definition"""
from abc import ABC, abstractmethod

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from mlkit.nn.confmodels import DatasetConf


class MLKitAbstractDataset(ABC, pl.LightningDataModule):
    def __init__(self, conf: DatasetConf):
        super().__init__()
        self.conf = conf
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        pass

    def prepare_traindataset(self, **kwargs) -> Dataset:
        """Prepare dataset for training.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

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

    def prepare_valdataset(self, **kwargs) -> Dataset:
        """Prepare dataset for validation.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

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

    def prepare_trainvaldataset(self, **kwargs) -> tuple[Dataset, Dataset]:
        """Prepare dataset for training and validation.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

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

    def prepare_testdataset(self, **kwargs) -> Dataset:
        """Prepare dataset for testing.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

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

    def prepare_predictdataset(self, **kwargs) -> Dataset:
        """Prepare dataset for predicting.

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the dataset

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

    def setup(self, stage: str):
        match stage:
            case "fit":
                if self.conf.trainval:
                    self.train_dataset, self.val_dataset = (
                        self.prepare_trainvaldataset(
                            **self.conf.trainval.arguments
                        )
                    )
                else:
                    self.train_dataset = self.prepare_traindataset(
                        **self.conf.train.arguments
                    )
                    self.val_dataset = self.prepare_valdataset(
                        **self.conf.train.arguments
                    )
            case "test":
                assert self.conf.test, (
                    "`test_config` is not defined. did you forget to define"
                    " `[dataset.test]` section in the configuration file?"
                )
                self.val_dataset = self.prepare_testdataset(
                    **self.conf.test.arguments
                )
            case "predict":
                assert self.conf.predict, (
                    "`test_config` is not defined. did you forget to define"
                    " `[dataset.predict]` section in the configuration file?"
                )
                self.predict_dataset = self.prepare_predictdataset(
                    **self.conf.predict.arguments
                )

    def train_dataloader(self):
        assert self.train_dataset is not None, (
            "`train_dataset` must be set before calling `train_dataloader`"
            " method!"
        )
        return DataLoader(self.train_dataset, **self.conf.train.loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.conf.validation.loader)

    def test_dataloader(self):
        assert self.conf.test, "configuration file was not defined for TESTING"
        return DataLoader(self.test_dataset, **self.conf.test.loader)

    def predict_dataloader(self):
        assert (
            self.conf.predict
        ), "configuration file was not defined for PREDICTION"
        return DataLoader(self.predict_dataset, **self.conf.predict.loader)

    def numpy_train_dataloader(self):
        pass

    def numpy_val_dataloader(self):
        pass

    def numpy_testdataloader(self):
        pass
