# NOTE: we can import local modules as project directory (the directory
# where the conf.toml file is located) is added to the path
import glob
import os

import h5py
import s3dis  # NOTE: local module s3dis.py
from torch import Tensor
from torch.utils.data import Dataset

from mlkit import MLKitAbstractDataModule


class S3DISDataset(Dataset):
    """S3DIS PyTorch dataset."""

    __slots__ = ("files",)

    def __init__(self, files: list[str]) -> None:
        super().__init__()
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def load_hdfs_pt(file: str):
        with h5py.File(file, "r") as h5:
            return (
                h5[s3dis.HDF5_FEATURES_KEY][:],
                h5[s3dis.HDF5_LABELS_KEY][:],
                h5[s3dis.HDF5_INSTANCE_KEY][:],
            )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        features, labels = S3DISDataset.load_hdfs_pt(self.files[idx])
        return (features, labels)


class S3DISDatamodule(MLKitAbstractDataModule):
    """Datamodule managing S3DIS dataset."""

    def prepare_data(self):
        """Convert raw S3DIS dataset to area HDF5 files.

        Assuming we have raw S3DIS dataset in the TXT files,
        we want to convert it to HDF5 files (one per each room).
        """
        # NOTE: DO NOT set state inside this method!
        # NOTE: fast check to skip data collection
        if s3dis.count_hdf5_rooms() == s3dis.S3DIS_ROOM_NBR:
            # NOTE: you can use logger-like methods for
            # MLKitAbstractDataModule subclasses
            self.info("all room files are ready. skipping!")
            return
        for area_id in range(1, s3dis.AREAS_NBR + 1):
            self.info("processing Area_%d", area_id)
            s3dis.prepare_single_area(area_id=area_id)

    def prepare_traindataset(self, test_area: int) -> Dataset:
        train_files = [
            file
            for file in s3dis.hdf5_files()
            if f"Area_{test_area}" not in file
        ]
        return S3DISDataset(files=train_files)

    def prepare_valdataset(self, test_area: int) -> Dataset:
        test_files = [
            file for file in s3dis.hdf5_files() if f"Area_{test_area}" in file
        ]
        return S3DISDataset(files=test_files)
