"""Kit4DL package."""
from lightning.pytorch.callbacks import Callback as Kit4DLCallback

from kit4dl._version import __version__
from kit4dl.dataset import Kit4DLAbstractDataModule
from kit4dl.nn.base import Kit4DLAbstractModule, StepOutput

# NOTE: aliases for backward compatibility with the version 2023.08b0
MLKitAbstractDataModule = Kit4DLAbstractDataModule
MLKitAbstractModule = Kit4DLAbstractModule
