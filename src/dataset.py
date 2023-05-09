from abc import ABC, abstractmethod

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.nn.confmodels import DatasetConfig


class AbstractDataset(ABC, pl.LightningDataModule):
    def __init__(
        self,
        train_config: DatasetConfig,
        val_config: DatasetConfig | None = None,
        test_config: DatasetConfig | None = None,
    ):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: str):
        raise NotImplementedError

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
        assert (
            self.val_dataset is not None
        ), "`val_dataset` must be set before calling `val_dataloader` method!"
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            shuffle=self.val_config.shuffle,
            num_workers=self.val_config.num_workers,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, (
            "`test_dataset` must be set before calling `test_dataloader`"
            " method!"
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            shuffle=self.test_config.shuffle,
            num_workers=self.test_config.num_workers,
        )

    def numpy_train_dataloader(self):
        pass

    def numpy_val_dataloader(self):
        pass

    def numpy_testdataloader(self):
        pass
