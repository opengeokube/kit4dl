"""Kit4DL package."""
from kit4dl.dataset import Kit4DLAbstractDataModule
from kit4dl.nn.base import Kit4DLAbstractModule

# NOTE: aliases for backward compatibility with the version 2023.08b0
MLKitAbstractDataModule = Kit4DLAbstractDataModule
MLKitAbstractModule = Kit4DLAbstractModule
