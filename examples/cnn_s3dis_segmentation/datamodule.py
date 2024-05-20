# NOTE: we can import local modules as project directory (the directory
# where the conf.toml file is located) is added to the path
import h5py
import numpy as np
import s3dis  # NOTE: local module s3dis.py
from torch.utils.data import Dataset

from kit4dl import Kit4DLAbstractDataModule


class S3DISDataset(Dataset):
    """S3DIS PyTorch dataset."""

    __slots__ = ("files",)

    def __init__(self, files: list[str], max_pts: int = 3000000) -> None:
        super().__init__()
        self.files = files
        self.max_pts = max_pts

    def _limit_points(self, features, labels, instances):
        assert (
            len(features) == len(labels) == len(instances)
        ), "shapes mismach!"
        if len(features) <= self.max_pts:
            return (features, labels, instances)
        idx = np.arange(len(features))
        np.random.shuffle(idx)
        idx = idx[: self.max_pts]
        return (features[idx], labels[idx], instances[idx])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        features, labels, instances = S3DISDataset.load_hdfs_pt(
            self.files[index]
        )
        features, labels, instances = self._limit_points(
            features, labels, instances
        )
        return (features, labels, instances)

    @staticmethod
    def load_hdfs_pt(file: str):
        with h5py.File(file, "r") as h5:
            return (
                h5[s3dis.HDF5_FEATURES_KEY][:],
                h5[s3dis.HDF5_LABELS_KEY][:].astype(np.int64),
                h5[s3dis.HDF5_INSTANCE_KEY][:].astype(np.int64),
            )


class S3DISDatamodule(Kit4DLAbstractDataModule):
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
            # Kit4DLAbstractDataModule subclasses
            self.info("all room files are ready. skipping!")
            return
        for area_id in range(1, s3dis.AREAS_NBR + 1):
            self.info("processing Area_%d", area_id)
            s3dis.prepare_single_area(area_id=area_id)

    def prepare_traindatasets(self, test_area: int) -> Dataset:
        train_files = [
            file
            for file in s3dis.hdf5_files()
            if f"Area_{test_area}" not in file
        ]
        return S3DISDataset(files=train_files)

    def prepare_valdatasets(self, test_area: int) -> Dataset:
        test_files = [
            file for file in s3dis.hdf5_files() if f"Area_{test_area}" in file
        ]
        return S3DISDataset(files=test_files)
