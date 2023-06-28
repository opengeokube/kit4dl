<div align="center">
<img src="static/logo.svg">

# A quick way to start with machine and deep learning
[![python](https://img.shields.io/badge/-Python_3.10%7C3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads)

[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) 
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)

[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/opengeokube/ml-kit/blob/main/LICENSE)

[![pytest](https://github.com/opengeokube/ml-kit/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/opengeokube/ml-kit/actions/workflows/test.yml)
</div>

## 🖋️ Authors
OpenGeokube Developers:
1. Jakub Walczak <a href="https://orcid.org/0000-0002-5632-9484"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>
1. Marco Macini <a href="https://orcid.org/0000-0002-9150-943X"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>
1. Mirko Stojiljkovic <a href="https://orcid.org/0000-0003-2256-1645"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>
1. Shahbaz Alvi <a href="https://orcid.org/0000-0001-5779-8568"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

## 📜 Cite Us
```bibtex
@ONLINE{ml-kit,
  author = {Walczak, J., Mancini, M., Stojiljkovic, M., Alvi, S.},
  title = {{MLKit}: A quick way to start with machine and deep learning},
  year = 2023,
  url = {https://github.com/opengeokube/ml-kit},
  urldate = {<access date here>}
}
```
## 🚧 Roadmap
- [ ] add handling sklearn-like models
- [ ] add functionality to serve the model
- [ ] write more unit tests


## 🎬 Quickstart

### Getting started

#### Installation
```bash
pip install mlkit
```

or

```bash
conda install ...
```

For contributing:
 
```text
git clone https://github.com/opengeokube/ml-kit
cd ml-kit
conda env create -f dev-env.yaml
pip install .
```

#### Preparing simple project
To start the new project in the current working directory, just run the following command:

```bash
mlkit init --name=my-new-project
```

It will create a directory with the name `my-new-project` where you'll find sample files.
Implement necessery methods for datamodule (`dataset.py`) and network (`model.py`).
Then, adjust `conf.toml` according to your needs. 
That's all 🎉

#### Running the training
To run the training just type the following command:

```bash
mlkit train
```

If the `conf.toml` file is present in your current working directory, the training will start.

If you need to specify the path to the configuration file, use `--conf` argument:
```bash
mlkit train --conf=/path/to/your/conf.toml
```

#### Serving the model
The packuge does not yet support model serving.


## 💡 Instruction
1. [Configuring base setup](#configuring-base-setup)
1. [Defining model](#defining-model)
1. [Defining datamodule](#defining-datamodule)
1. [Defining training](#defining-training)
1. [Configuring optimizer](#configuring-optimizer)
1. [Configuring criterion](#configuring-criterion)
1. [Configuring metrics](#configuring-metrics)
1. [Defining `target`](#defining-target)
1. [Substitutable symbols](#substitutable-symbols)

#### Configuring base setup
Most of the training/validation procedure is managed by a configuration file in the TOML format (reccomended name is `conf.toml`).
Each aspect is covered by separate sections. The general one is called `[base]`.
It has the following properties:

|   **Property** 	|  **Type**        |                                                   **Details**                                          	|
|---------------	|----------------- | -------------------------------------------------------------------------------------------------------	| 
|      `seed`	    |   `int`          |  seed of the random numbers generators for `NumPy` and `PyTorch`                                       	| 
|     `cuda_id`   |  `int` or `None` |  ID of the cuda device (if available) or `None` for `CPU`                                                |
|`experiment_name`| `str`            |  name of the experiment                                                                                  |
|   `log_level`   |  `str`           | logging level, should be one out of `debug`, `info`, `warn`, `error`, `critical`                         |
|  `log_format`   | `str`            | format of the logging message accoridng to Python's `logging` package  (e.g. `"%(asctime)s - %(name)s`)  |


##### ✍️ Example
```toml
[base]
seed = 0
experiment_name = "point_clout_segmentation"
log_level = "info"
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### Defining model
The machine learning/deep learning model definition is realized in two aspects. 
1. The definition of the model (e.g. PyTorch model) in the `.py` file.
1. The configuration in the `[model]` section of the configuration file.

The file with the model definition should contain a subclass of `MLKitAbstractModule` abstract class of the `mlkit` package.
The subclass should implement, at least, abstract methods `configure` and `run_step`.
In the `configure` method, the architecture of the network should be defined. 
In `run_step` method, it turn, the logic for single forward pass should be implemented.

##### ✍️ Example
```python
import torch
from torch import nn
from mlkit import MLKitAbstractModule

class SimpleCNN(MLKitAbstractModule):
    def configure(self, input_dims, layers, dropout, output_dims) -> None:
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_dims, 16, kernel_size=3, padding="same", bias=True
            ),
            nn.ReLU(),
        )

    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, label = batch
        return label, self.l1(x)
```
> ❗ Note that `run_step` method should return a tuple of two tensors, the ground-truth labels and the output of the network.

> ❗ Note that `batch` argument can be unpacked depending on how you define your dataset for datamodule (see [Defining datamodule](#defining-datamodule))

In the configuration file, in the dedicated `[model]` section, at least `target` property should be set. The extra arguments are treated as the arguments for the `configure` method.

> ❗ Note that arguments' values of the `configure` method (i.e. `input_dims`, `layers`, `dropout`, and `output_dims`) are taken from the configuration files. Those names can be arbitrary.

```toml
[model]
target = "dgcnn::DGCNN"
input_dims = 1
layers = 4
dropout = 0.5
output_dims = 10
```
> ❗ `target` is a required parameter that **must** be set. It contains a path to the class (a subclass of `MLKitAbstractModule`). To learn how `target` could be defined, see Section [Defining `target`](#defining-target).

If a forward pass for your model differs for the training, validation, test, or prediction stages, you can define separate methods for them:
```python
import torch
from torch import nn
from mlkit import MLKitAbstractModule

class SimpleCNN(MLKitAbstractModule):
    ...
    def run_val_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def run_test_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def run_predict_step(self, batch, batch_idx) -> torch.Tensor:
        pass            
```

> ❗ If you need more customization of the process, you can always override the existing methods according to your needs.

#### Defining datamodule
Similarily to the model, datamodule instance is fully defined by the Python class and its configuration.
The datamodule need to be a subclass of the `MLKitAbstractDataModule` abstract class from the `mlkit` package.
The class has to implement, at least, `prepare_trainvaldataset` (if preparing is the same for the train and validation splits) or `prepare_traindataset` and `prepare_valdataset` (if preparing data differs). Besides those, you can define `prepare_testdataset` and `prepare_predictdataset`, for test and prediction, respectively.
```python
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from mlkit import MLKitAbstractDataModule


class MNISTCustomDatamodule(MLKitAbstractDataModule):
    def prepare_trainvaldataset(
        self, root_dir: str
    ) -> tuple[Dataset, Dataset]:
        dset = MNIST(
            root=root_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        train_dset, val_dset = random_split(dset, [0.8, 0.2])
        return (train_dset, val_dset)

    def prepare_testdataset(self, root_dir: str) -> Dataset:
        return MNIST(
            root=root_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
```

If you need to acquire data or do some other processing, implement `prepare_data` method.

```python
...
class MNISTCustomDatamodule(MLKitAbstractDataModule):
    ...
    def prepare_data(self):
        ...
```

> ❗ **DO NOT** set state inside `prepare_data` method (~~`self.x = ...`~~).

If you need more customization, feel free to override the other methods of `MLKitAbstractDataModule` superclass.

In the configuration file, there are dedicated `[dataset]`-related sections.

```toml
[dataset]
target = "./datamodule.py::MNISTCustomDatamodule"

[dataset.trainval]
root_dir = "./mnist"

[dataset.train.loader]
batch_size = 150
shuffle = true
num_workers = 4

[dataset.validation.loader]
batch_size = 150
shuffle = false
num_workers = 4
```

In the root `[dataset]` you should define `target` property being a path to the subclass of the `MLKitAbstractDataModule` module (see [Defining `target`](#defining-target)).
Then, you need to define either `[dataset.trainval]` section or two separate sections: `[dataset.train]`, `[dataset.validation]`. There are also optional sections: `[dataset.test]` and `[dataset.predict]`.
In `[dataset.trainval]` you pass values for parameters of the `prepare_trainvaldataset` method.
Respectively, in the `[dataset.train]` you pass values for the parameters of the `prepare_traindataset` method, in `[dataset.validation]` — `prepare_valdataset`, `[dataset.test]` — `prepare_testdataset`, `[dataset.predict]` — `prepare_predictdataset`.

Besides dataset configuration, you need to specify data loader arguments as indicated in the PyTorch docs [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

> ❗ You **cannot** specify loader arguments for in the `[dataset.trainval.loader]`. Loaders should be defined for each split separately.

#### Defining training

#### Configuring optimizer

#### Configuring criterion

#### Configuring metrics

#### Configuring checkpoint

#### Defining `target`

#### Substitutable symbols
In the configuration file you can use symbols that will be substituted during the runtime.
The symbols should be used in curly brackets (e.g. `{PROJECT_DIR}`.)

|   **Symbol** 	|            **Meaning of the symbol**                                   	|          **Example**                |
|-------------	|-----------------------------------------------------------------------	| -----------------------------------	|
| `PROJECT_DIR`	| the home directory of the TOML configuration file (project directory) 	| `target = {PROJECT_DIR}/model.py`



