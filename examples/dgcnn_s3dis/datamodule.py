# NOTE: we can import local modules as project directory (the directory
# where the conf.toml file is located) is added to the path
import s3dis
from torch import Tensor
from torch.utils.data import Dataset

from mlkit import MLKitAbstractDataModule


class S3DISDataset(Dataset):
    """S3DIS PyTorch dataset"""

    __slots__ = (
        "x",
        "y",
    )

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__()


class S3DISDatamodule(MLKitAbstractDataModule):
    """Datamodule managing S3DIS dataset."""

    def prepare_data(self):
        """Convert raw S3DIS dataset to area HDF5 files.

        Assuming we have raw S3DIS dataset in the TXT files,
        we want to convert it to HDF5 files (one per each room).
        """
        # NOTE: do not set state inside this method!
        # NOTE: fast check to skip data collection
        if s3dis.count_hdf5_rooms() == s3dis.S3DIS_ROOM_NBR:
            # NOTE: you can use logger-like methods for
            # MLKitAbstractDataModule subclasses
            self.info("all room files are ready. skipping!")
            return
        for area_id in range(1, s3dis.AREAS_NBR + 1):
            self.info("processing Area_%d", area_id)
            s3dis.prepare_single_area(area_id=area_id)

    def prepare_traindataset(self, s3dis_root_dir: str, test_area: int) -> Dataset:
        breakpoint()
        return S3DISDataset()

    def prepare_valdataset(self, s3dis_root_dir: str, test_area: int) -> Dataset:
        breakpoint()
        return S3DISDataset()
