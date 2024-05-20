# Specifying data module

::: kit4dl.nn.dataset.Kit4DLAbstractDataModule


## Custom splits

!!! Note
    Available since 2024.5b0

n Kit4DL, you can easily define the logic for cross-validation. Starting from the version 2024.5b0 the old method `prepare_trainvaldataset` was replaced by the `prepare_trainvaldatasets` method that is a generator. You define the logic of the generator by yourself. To run 10-fold cross validation, implement the method in the following way:

``` { .python .copy }
...
from sklearn.model_selection import KFold

class MNISTCustomDatamodule(Kit4DLAbstractDataModule):
    def prepare_trainvaldatasets(self, root_dir: str):
        dset = MNIST(
            root=root_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        split = KFold(n_splits=10, shuffle=True, random_state=0)
        for i, (train_ind, val_ind) in enumerate(
            split.split(dset.data, dset.targets)
        ):
            yield Subset(dset, train_ind), Subset(dset, val_ind)
```

If you want to stick to the old logic and return a single split, just `yield` the corresponding datasets:

``` { .python .copy }
...

class MNISTCustomDatamodule(Kit4DLAbstractDataModule):
    def prepare_trainvaldatasets(self, root_dir: str):
        tr_dset = MNIST(
            root=root_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        ts_dset = MNIST(
            root=root_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )        
        yield tr_dset, ts_dset
```

Each generated tuple of train and validation dataset will be fed into the training/validation pipeline. If you use external metric loggers, results for each split will be uploaded using the experiment name and the suffix like `(split=0)`.

The suffix can be overwriten by the environmental variable `KIT4DL_SPLIT_PREFIX`.        