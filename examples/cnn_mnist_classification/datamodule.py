from torch.utils.data import Dataset, TensorDataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from sklearn.model_selection import KFold
from kit4dl import Kit4DLAbstractDataModule


class MNISTCustomDatamodule(Kit4DLAbstractDataModule):
    def prepare_trainvaldatasets(self, root_dir: str):
        dset = MNIST(
            root=root_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        split = KFold(n_splits=5, shuffle=True, random_state=0)
        for i, (train_ind, val_ind) in enumerate(
            split.split(dset.data, dset.targets)
        ):
            yield Subset(dset, train_ind), Subset(dset, val_ind)

    def prepare_testdataset(self, root_dir: str):
        return MNIST(
            root=root_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
