"""This is an extra module separated from the `dataset.py` for clarity.

The module contains functions to prepare Area HDF5 files to accelerate
loading and processing.
"""
import glob
import os
from typing import Iterator

try:
    import h5py
except ModuleNotFoundError as err:
    raise RuntimeError(
        "did you forget to install `h5py` package? run `conda install h5py` or"
        " `pip install h5py`"
    )

try:
    from tqdm import tqdm
except ModuleNotFoundError as err:
    raise RuntimeError(
        "did you forget to install `tqdm` package? run `conda install -c"
        " conda-forge tqdm` or `pip install tqdm`"
    )

import numpy as np
import pandas as pd

from kit4dl import context

SEP = " "
AREAS_NBR = 6
COLUMN_NAMES = ["x", "y", "z", "r", "g", "b"]
LABEL_COLUMN = "L"
INSTANCE_COLUMNT = "I"
COLUMN_NAMES_LABEL = COLUMN_NAMES + [LABEL_COLUMN, INSTANCE_COLUMNT]
CLASS_NAME_TO_LABEL = {
    "beam": 0,
    "board": 1,
    "bookcase": 2,
    "ceiling": 3,
    "chair": 4,
    "clutter": 5,
    "stairs": 5,  # stairs are treated as clutter
    "column": 6,
    "door": 7,
    "floor": 8,
    "sofa": 9,
    "stairs": 10,
    "table": 11,
    "wall": 12,
    "window": 13,
}
S3DIS_ROOM_NBR = 273
HDF5_FEATURES_KEY = "pts"
HDF5_LABELS_KEY = "labels"
HDF5_INSTANCE_KEY = "instance"


def base_hdf5_dir() -> str:
    # NOTE: DO NOT use context attributes on the module level
    # if you need them, use function wrapper.
    # Context attributes are populated after importing
    return os.path.join(context.PROJECT_DIR, "data", "hdf5")


def _get_rooms_names(area_id: int) -> list[str]:
    # NOTE: avoid passing relative path either use `kit4dl.session` attributes
    # or pass absolute path directly
    return glob.glob(
        os.path.join(context.PROJECT_DIR, "data", f"Area_{area_id}", "*")
    )


def _get_files_for_room_path(room_path: str) -> Iterator[str]:
    yield from glob.iglob(os.path.join(room_path, "Annotations", "*.txt"))


def _get_label_from_path(path: str) -> int:
    return CLASS_NAME_TO_LABEL[path.split(os.sep)[-1].split("_")[0]]


def instance_id_generator():
    ins_id: int = 0
    while True:
        yield ins_id
        ins_id += 1


def _concat_pts_and_labels(
    pts: pd.DataFrame, obj_cls_label: int, obj_ins_label: int
) -> pd.DataFrame:
    return pts.assign(
        **{
            LABEL_COLUMN: np.repeat(obj_cls_label, len(pts)),
            INSTANCE_COLUMNT: np.repeat(obj_ins_label, len(pts)),
        }
    )


def _save_pt_to_hdf5(pt: pd.DataFrame, fname: str) -> None:
    with h5py.File(fname, "w") as file:
        file.create_dataset(
            HDF5_FEATURES_KEY, data=pt[COLUMN_NAMES].astype(np.float32).values
        )
        file.create_dataset(
            HDF5_LABELS_KEY, data=pt[LABEL_COLUMN].astype(np.int32).values
        )
        file.create_dataset(
            HDF5_INSTANCE_KEY,
            data=pt[INSTANCE_COLUMNT].astype(np.int32).values,
        )


def _get_res_hdf5_path(room_path: str) -> str:
    base_dir = base_hdf5_dir()
    os.makedirs(base_dir, exist_ok=True)
    path_components = room_path.split(os.sep)
    room_area_name = "_".join(path_components[-2:])
    return os.path.join(base_dir, f"{room_area_name}.h5")


def prepare_single_area(area_id: int, overwrite: bool = False) -> None:
    """Prepare HDF5 files for room in Area `area_id`.

    If room HDF5 file exists, it will be skipped unless `overwrite` flag
    it set to `True`.

    Parameters
    ----------
    area_id : int
        Area of S3DIS dataset to process
    overwrite : bool, default `False`
        If existing files should be overwritten
    """
    pbar = tqdm(_get_rooms_names(area_id=area_id))
    for room_path in pbar:
        inst_id_gen = instance_id_generator()
        pbar.set_description(f"processing room: {room_path}", refresh=True)
        hdf5_room_path = _get_res_hdf5_path(room_path)
        if os.path.exists(hdf5_room_path) and (not overwrite):
            continue
        room_pts: pd.DataFrame = pd.DataFrame(columns=COLUMN_NAMES_LABEL)
        for abs_obj_file in _get_files_for_room_path(room_path=room_path):
            obj_cls_label = _get_label_from_path(abs_obj_file)
            obj_ins_label = next(inst_id_gen)
            obj_pts = pd.read_csv(abs_obj_file, sep=SEP, names=COLUMN_NAMES)
            obj_pts = _concat_pts_and_labels(
                obj_pts,
                obj_cls_label=obj_cls_label,
                obj_ins_label=obj_ins_label,
            )
            room_pts = pd.concat([room_pts, obj_pts])
        _save_pt_to_hdf5(room_pts, hdf5_room_path)


def hdf5_files() -> list[str]:
    """Get all HDF5 files for S3DIS."""
    return glob.glob(os.path.join(base_hdf5_dir(), "*.h5"))


def count_hdf5_rooms() -> int:
    """Return the number of HDF5 files for S3DIS dataset."""
    return len(hdf5_files())
