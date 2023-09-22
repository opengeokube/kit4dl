<div align="center">
<img src="https://raw.githubusercontent.com/opengeokube/ml-kit/56dc56c1be7f6332c0f75cdfb3160d29cebc3c58/static/logo.svg" width="20%" height="20%">

# A quick way to start with machine and deep learning
[![python](https://img.shields.io/badge/-Python_3.10%7C3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads)

[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) 
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)

[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/opengeokube/kit4dl/blob/main/LICENSE)

[![pytest](https://github.com/opengeokube/kit4dl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/opengeokube/kit4dl/actions/workflows/test.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8328176.svg)](https://doi.org/10.5281/zenodo.8328176)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/kit4dl.svg)](https://anaconda.org/conda-forge/kit4dl) 
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/kit4dl.svg)](https://anaconda.org/conda-forge/kit4dl)

[![PyPI version](https://badge.fury.io/py/kit4dl.svg)](https://badge.fury.io/py/kit4dl)
</div>




## 🖋️ Authors
OpenGeokube Developers:
1. Jakub Walczak <a href="https://orcid.org/0000-0002-5632-9484"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>
1. Marco Macini <a href="https://orcid.org/0000-0002-9150-943X"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>
1. Mirko Stojiljkovic <a href="https://orcid.org/0000-0003-2256-1645"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>
1. Shahbaz Alvi <a href="https://orcid.org/0000-0001-5779-8568"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

## 📜 Cite Us
```bibtex
@SOFTWARE{kit4dl,
  author = {Walczak, Jakub and
            Mancini, Marco and
            Stojiljković, Mirko and
            Alvi, Shahbaz},
  title = {Kit4DL},
  month = sep,
  year = 2023,
  note = {{Available in GitHub: https://github.com/opengeokube/kit4dl}},
  publisher = {Zenodo},
  version = {2023.9b1},
  doi = {10.5281/zenodo.8328176},
  url = {https://doi.org/10.5281/zenodo.8328176}
}
```
## 🚧 Roadmap

> **Warning**: Kit4DL is currently in its alpha stage. All recommendations are welcomed.

- [ ] add handling sklearn-like models
- [ ] add functionality to serve the model
- [x] enable custom metrics
- [x] enable using callbacks (also custom ones)
- [ ] write more unit tests


## 🎬 Quickstart

### Getting started

#### Installation
```bash
pip install kit4dl
```

or

```bash
conda install -c conda-forge kit4dl 
```

For contributing:

Download and install the `make` tool unless it is already available in your system. 

```text
git clone https://github.com/opengeokube/kit4dl
cd kit4dl
conda env create -f dev-env.yaml
pip install -e .
```

#### Preparing simple project
To start the new project in the current working directory, just run the following command:

```bash
kit4dl init --name=my-new-project
```

It will create a directory with the name `my-new-project` where you'll find sample files.
Implement necessery methods for datamodule (`dataset.py`) and network (`model.py`).
Then, adjust `conf.toml` according to your needs. 
That's all 🎉

#### Running the training
To run the training just type the following command:

```bash
kit4dl train
```
> **Note**: If you want to run also test for best saved weight, use flag `--test`


If the `conf.toml` file is present in your current working directory, the training will start.

If you need to specify the path to the configuration file, use `--conf` argument:
```bash
kit4dl train --conf=/path/to/your/conf.toml
```

#### Serving the model
The packuge does not yet support model serving.

## 🪁 Playground
At first, install `kit4dl` package as indicated in the Section [Installation](#installation).

#### Handwritten digit recognition
Just navigate to the directory `/examples/cnn_mnist_classification` and run
```bash
kit4dl train
```

#### Point cloud instance segmentation
Just navigate to the directory `/examples/cnn_s3dis_segmentation` and run
```bash
kit4dl train
```


## 💡 Instruction
1. [Configuring base setup](#configuring-base-setup)
1. [Configuring logging](#configuring-logging)
1. [Defining model](#defining-model)
1. [Defining datamodule](#defining-datamodule)
1. [Configuring training](#configuring-training)
1. [Configuring optimizer](#configuring-optimizer)
1. [Configuring criterion](#configuring-criterion)
1. [Configuring metrics](#configuring-metrics)
1. [Configuring checkpoint](#configuring-checkpoint)
1. [Defining `target`](#defining-target)
1. [Substitutable symbols](#substitutable-symbols)
1. [Context constants](#context-constants)

#### Configuring base setup
Most of the training/validation procedure is managed by a configuration file in the TOML format (recommended name is `conf.toml`).
Each aspect is covered by separate sections. The general one is called `[base]`.
It has the following properties:

|   **Property** 	|  **Type**        |                                                   **Details**                                          	|
|---------------	|----------------- | -------------------------------------------------------------------------------------------------------	| 
|      `seed`	    |   `int`          |  seed of the random numbers generators for `NumPy` and `PyTorch`                                       	| 
|     `cuda_id`     |  `int` or `None` |  ID of the cuda device (if available) or `None` for `CPU`                                                |
|`experiment_name`* | `str`            |  name of the experiment                                                                                  |

> **Note**: Arguments marked with `*` are obligatory!

> **Warning**: Remember to install the version of `pytorch-cuda` package compliant to your CUDA Toolkit version.


##### ✍️ Example
```toml
[base]
seed = 0
cuda_id = 1
experiment_name = "point_clout_segmentation"
```

#### Configuring logging
Logging section is optional but it provides you with some extra flexibility regarding the logging.
All configuration related to logging is included in the `[logging]` section of the configuration file. 
You can define following properties:

|   **Property** 	|  **Type**        |                                                   **Details**                                          	|
|---------------	|----------------- | -------------------------------------------------------------------------------------------------------	| 
|      `type`	    |   `str`          |  type of metric logger (one of the value: `"comet"`, `"csv"`, `"mlflow"`, `"neptune"`, `"tensorboard"`, `"wandb"` - metric loggers supported by PyTorch Lightning [https://lightning.ai/docs/pytorch/stable/api_references.html#loggers](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers). **DEFAULT:** `csv`)                                       	| 
|     `level`     |  `str` |  Python-supported logging levels (i.e. `"DEBUG"`, `"INFO"`, `"WARN"`, `"ERROR"`, `"CRITICAL"`)  **DEFAULT:** `INFO`                                               |
|`format` | `str`            |  logging message format as defined for the Python `logging` package (see [https://docs.python.org/3/library/logging.html#logging.LogRecord](https://docs.python.org/3/library/logging.html#logging.LogRecord))                                                                               |

> **Warning**: Logger `level` and `format` are related to the Python `logging` Loggers you can use in your model and datamodule classes with approperiate methods `self.debure`, `self.info`, etc. In `type`, in turn, you just specify the metric logger as used in PyTorch Lightning package!

> **Note**: All required arguments for metric logger can be specified as extra arguments in the `[logging]`section.

##### ✍️ Example
```toml
[logging]
# we gonna use CSVLogger
type = "csv"
# for CSVLogger, we need to define 'save_dir' argument and/or
# other extra ones (https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.csv_logs.html#module-lightning.pytorch.loggers.csv_logs)
save_dir = "{{ PROJECT_DIR }}/my_metrics.csv"

# then we define level and format for logging messages
level = "info"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

> **Note**: If you don't pass a `name` or `experiment_name` argument explicitly for the metric logger, the `experiment_name` value defined in the `[base]` section will be applied as, respectively: `name` argument for `csv`, `neptune`, `tensorboard`, `wandb`, and as `experiment_name` for `comet` and `mlflow`.


#### Defining model
The machine learning/deep learning model definition is realized in two aspects. 
1. The definition of the model (e.g. PyTorch model) in the `.py` file.
1. The configuration in the `[model]` section of the configuration file.

The file with the model definition should contain a subclass of `Kit4DLAbstractModule` abstract class of the `kit4dl` package.
The subclass should implement, at least, abstract methods `configure` and `run_step`.
In the `configure` method, the architecture of the network should be defined. 
In `run_step` method, it turn, the logic for single forward pass should be implemented.

##### ✍️ Example
```python
import torch
from torch import nn
from kit4dl import Kit4DLAbstractModule

class SimpleCNN(Kit4DLAbstractModule):
    def configure(self, input_dims, output_dims) -> None:
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_dims, 16, kernel_size=3, padding="same", bias=True
            ),
            nn.ReLU(),
        )

    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
        x, label = batch
        logits = self.l1(x)
        preds = logits.argmax(dim=-1)
        return label, logits, preds
```
> **Note**: `run_step` method should return a tuple of 2 (ground-truth, scores) or 3 (ground-truth, scores, loss) tensors.

> **Note**: `batch` argument can be unpacked depending on how you define your dataset for datamodule (see [Defining datamodule](#defining-datamodule))

In the configuration file, in the dedicated `[model]` section, at least `target` property should be set. The extra arguments are treated as the arguments for the `configure` method.

> **Note**: Arguments' values of the `configure` method (i.e. `input_dims` and `output_dims`) are taken from the configuration files. Those names can be arbitrary.

##### ✍️ Example
```toml
[model]
target = "./model.py::SimpleCNN" 
input_dims = 1
output_dims = 10
```
> **Note**: `target` is a required parameter that **must** be set. It contains a path to the class (a subclass of `Kit4DLAbstractModule`). To learn how `target` could be defined, see Section [Defining `target`](#defining-target).

If a forward pass for your model differs for the training, validation, test, or prediction stages, you can define separate methods for them:

##### ✍️ Example
```python
import torch
from torch import nn
from kit4dl import Kit4DLAbstractModule

class SimpleCNN(Kit4DLAbstractModule):
    ...
    def run_val_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def run_test_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def run_predict_step(self, batch, batch_idx) -> torch.Tensor:
        pass            
```

> **Note**: If you need more customization of the process, you can always override the existing methods according to your needs.

#### Defining datamodule
Similarily to the model, datamodule instance is fully defined by the Python class and its configuration.
The datamodule need to be a subclass of the `Kit4DLAbstractDataModule` abstract class from the `kit4dl` package.
The class has to implement, at least, `prepare_trainvaldataset` (if preparing is the same for the train and validation splits) or `prepare_traindataset` and `prepare_valdataset` (if preparing data differs). Besides those, you can define `prepare_testdataset` and `prepare_predictdataset`, for test and prediction, respectively.

##### ✍️ Example
```python
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from kit4dl import Kit4DLAbstractDataModule


class MNISTCustomDatamodule(Kit4DLAbstractDataModule):
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

If you need to acquire data or do some other processing, implement `prepare_data` method. In that method you can use extra attributes you defined in the `[dataset]` section of the configuration file.

##### ✍️ Example
```toml
[dataset]
target = "./datamodule.py::MNISTCustomDatamodule"
my_variable = 10
```

```python
...
class MNISTCustomDatamodule(Kit4DLAbstractDataModule):
    my_variable: int # NOTE: To make attribute visible, we can declare it here

    def prepare_data(self):
        result = self.my_variable * 2
```

> **Warning**: **DO NOT** set state inside `prepare_data` method (~~`self.x = ...`~~).

If you need more customization, feel free to override the other methods of `Kit4DLAbstractDataModule` superclass.
To force custom batch collation, override selected methods out of the following ones. They should return the proper callable object!

```python
def some_collate_func(samples: list): ...

class MNISTCustomDatamodule(Kit4DLAbstractDataModule):
    ...
    def get_train_collate_fn(self):
        return some_collate_func

    def get_val_collate_fn(self):
        return some_collate_func

    def get_test_collate_fn(self):
        return some_collate_func

    def get_predict_collate_fn(self):
        return some_collate_func
```

> **Warning**: **DO NOT** use nested function as a callation callable. It will fail due to pickling nested function error.

If you need a custom batch collation but the same for each stage (train/val/test/predict), implement the method `get_collate_fn()`:
```python
def get_collate_fn(self):
    return some_collate_func
```

In the configuration file, there are dedicated `[dataset]`-related sections.

##### ✍️ Example
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

In the root `[dataset]` you should define `target` property being a path to the subclass of the `Kit4DLAbstractDataModule` module (see [Defining `target`](#defining-target)).
Then, you need to define either `[dataset.trainval]` section or two separate sections: `[dataset.train]`, `[dataset.validation]`. There are also optional sections: `[dataset.test]` and `[dataset.predict]`.
In `[dataset.trainval]` you pass values for parameters of the `prepare_trainvaldataset` method.
Respectively, in the `[dataset.train]` you pass values for the parameters of the `prepare_traindataset` method, in `[dataset.validation]` — `prepare_valdataset`, `[dataset.test]` — `prepare_testdataset`, `[dataset.predict]` — `prepare_predictdataset`.

Besides dataset configuration, you need to specify data loader arguments as indicated in the PyTorch docs [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

> **Warning**: You **cannot** specify loader arguments for in the `[dataset.trainval.loader]`. Loaders should be defined for each split separately.


#### Configuring training
Training-related arguments should be defined in the `[training]` section of the configuration file.
You can define the following arguments.

|   **Property** 	|  **Type**        |         **Details**              |
|---------------	|----------------- | -------------------------------- | 
|      `epochs`*    |   `int > 0`      |  number of epochs	              | 
|      `callbacks`  |   `list`         |  list of callbacks	              | 
|`epoch_schedulers` |  `list of dict`  |  list of schedulers definitions  |

> **Note**: Arguments marked with `*` are obligatory!

You can define a list of custom callbacks applied in the training process. Your callbacks need to be subclasses of `lightning.pytorch.callbacks.Callback` or `kit4dl.Kit4DLCallback` (for convenience) class and define one/some of the methods indicated in the [PyTorch-Lightning callback API](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#callback-api). You can always use one of the [predefined callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#built-in-callbacks).


```toml
[training]
callbacks = [
    {target = "./callbacks.py::SaveConfusionMatrixCallback", task="multiclass", num_classes=10, save_dir="{{ PROJECT_DIR }}/cm},
    {target = "lightning.pytorch.callbacks::DeviceStatsMonitor"}
]
```

Where the 1st callback is user-defined and the other - PyTorch-Loghtning built-in. For the custom callback we need to provide a class (here: located in the `callbacks.py` file in the project directory, the class is named `SaveConfusionMatrixCallback`).

```python
import os
from typing import Any

import lightning.pytorch as pl
import torchmetrics as tm

from kit4dl import Kit4DLCallback, StepOutput


class SaveConfusionMatrixCallback(Kit4DLCallback):
    _cm: tm.ConfusionMatrix
    _num_classes: int
    _task: str
    _save_dir: str

    def __init__(self, task: str, num_classes: int, save_dir: str) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._save_dir = save_dir
        self._task = task
        os.makedirs(self._save_dir, exist_ok=True)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._cm = tm.ConfusionMatrix(
            task=self._task, num_classes=self._num_classes
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: StepOutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._cm.update(outputs.predictions, outputs.labels)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the val epoch ends."""
        fig, _ = self._cm.plot()
        target_file = os.path.join(
            self._save_dir,
            f"confusion_matrix_for_epoch_{pl_module.current_epoch}",
        )
        fig.savefig(target_file)

```






Besides those listed in the table above, you can specify PyTorch Lightning-related `Trainer` arguments, like:
1. `accumulate_grad_batches`
1. `gradient_clip_val`
1. `gradient_clip_algorithm`
1. ...

##### ✍️ Example

```toml
[training]
epochs = 10
epoch_schedulers = [
    {target = "torch.optim.lr_scheduler::CosineAnnealingLR", T_max = 100}
]
accumulate_grad_batches = 2
```

#### Configuring optimizer
Optimizer configuration is located in the subsection `[training.optimizer]`.
There, you should define `target` (see [Defining `target`](#defining-target)) and extra keyword arguments passed to the optimizer initializer.

##### ✍️ Example
```toml
[training.optimizer]
target = "torch.optim::Adam"
lr = 0.001
weight_decay = 0.01
```
> **Note**: The section `[training.optimizer]` is **mandatory**.
> **Note**: You can always define the custom optimizer. Then, you just need to set the proper `target` value.


#### Configuring criterion
Similarily to the optimizer configuration, there is a subsection dedicated for the critarion. 
You need to specify, at least, the `target` (see [Defining `target`](#defining-target)) and other mandatory or optional
properties of the selected critarion (loss function).

##### ✍️ Example
```toml
[training.criterion]
target = "torch.nn::CrossEntropyLoss"
weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
```

> **Note**: The section `[training.criterion]` is **mandatory**.
> **Note**: You can always define the custom optimizer. Then, you just need to set the proper `target` value.

#### Configuring metrics
Metrics are configured in the section `[metrics]` of the configuration file. You can define several metrics (including the custom ones). 
The only thing you need to do is to define all desired metrics. For each metric dictionary, you need to set `target` (see Section [Defining `target`](#defining-target)) value and, eventually, extra arguments. **REMEMBER** to have metric names (here `MyPrecision` and `FBetaScore`) unique!

##### ✍️ Example
```toml
[metrics]
MyPrecision = {target = "torchmetrics::Precision", task = "multiclass", num_classes=10}
FBetaScore = {target = "torchmetrics::FBetaScore", task = "multiclass", num_classes=10, beta = 0.1}
```
> **Note**: You can define custom metrics. Just properly set `target` value. **REMEMBER!** The custom metric need to be a subclass of `torchmetrics.Metric` class!

```python
import torch
import torchmetrics as tm

class MyMetric(tm.Metric):
    def __init__(self):
        ...
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ...
     def compute(self):
        ...
```

#### Configuring checkpoint
If you need to save your intermediate weights (do checkpoints) you can configure the optional subsection `[training.checkpoint]`.
In the section, you can define the following proeprties:

|   **Property** 	|  **Type**        |         **Details**              |
|---------------	|----------------- | -------------------------------- | 
|      `path`*      |   `str`          |    path to a directory where checkpoints should be stored	              | 
|`monitor`* |  `dict`  |  a dictionary with two keys: `metric` and `stage`. `metrics` is a metric name as defined in the `[metrics]` section ([Configuring metrics](#configuring-metrics)), `stage` is one of the following: [`train`, `val`]  |
|      `filename`*      |   `str`          |    filename pattern of the checkpoint (see (PyTorch Lightning `ModelCheckpoint`)[https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html])	you can use value of the defined metric for the stage. if you want `MyPrecision` score for the validation stage, use `{val_myprecision}` in the filename               | 
|      `mode`      |   `min` | `max`          |    to save checkpoint for `min`imum or `max`imum value of the metric being tracked (`monitor`). **default: `max`**	              | 
|      `save_top_k`      |   `int`          |   save checkepoints for the top `k` values of the metric. **default: `1`**	              |
|      `save_weights_only`      |   `bool`          |   if only weights should be saved (`True`) or other states (optimizer, scheduler) also (`False`). **default: `True`**	              |
|      `every_n_epochs`     |   `int`          |    The number of training epochs between saving sucessive checkpoints. **default: `1`**	       | 
|      `save_on_train_epoch_end`      |   `bool`          |    if `False` checkpointing is run at the end of the validation, otherwise - training   **default: `False`**	           | 

> **Note**: Arguments marked with `*` are obligatory!

##### ✍️ Example
```toml
[training.checkpoint]
path = "{{ PROJECT_DIR }}/chckpt"
monitor = {"metric" = "Precision", "stage" = "val"}
filename = "{epoch}_{val_precision:.2f}_cnn"
mode = "max"
save_top_k = 1
```

> **Note**: You can see we used substitutable symbol `{{ PROJECT_DIR }}`. More about them in the Section [Substitutable symbols](#substitutable-symbols).




#### Defining `target`
Target property in the Kit4DL package is kind of extended fully qualified name pointing to the classes supposed to use in the
given context, like for:
1. neural network class (`target = "./model.py::SimpleCNN"`)
1. datamodule (`target = "./datamodule.py::MNISTCustomDatamodule"`)
1. optimizer (`target = "torch.optim::Adam"`)
1. criterion (`target = "torch.nn::CrossEntropyLoss"`)
1. schedulers (`target = "torch.optim.lr_scheduler::CosineAnnealingLR"`)

> **Note**: As a package/module - class separator the double colon is used `::`!

It might be set in several different ways:
1. **By using a built-and installed package**. Then, you just need to specify the package/module name and the class name, like `target = "torch.nn::CrossEntropyLoss"`  (we use module `torch.nn` and class `CrossEntropyLoss` defined within).
1. **By using a custom module in the project directory**. The project directory, i.e. the directory where the confguration TOML file is located, is added to the `PYTHONPATH`, so you can freely use `.py` files defined there as modules. Having the module `model.py` with the `SimpleCNN` class definition, we can write `target` as `target = "model::SimpleCNN"`.
1. **By using a custom `.py` file.** In this case, you specify `target` as an absolute or relative (w.r.t. the configuration file) to a `.py` file, like `target = "./model.py::SimpleCNN"` or `target = "/usr/neural_nets/my_net/model.py::SimpleCNN"`.

> **Note**: For `target` definition you can use substitutable symbols defined below.

#### Substitutable symbols
In the configuration file you can use symbols that will be substituted during the runtime.
The symbols should be surrended by single spaces and in double curly brackets (e.g. `{{ PROJECT_DIR }}`.)

|   **Symbol** 	|            **Meaning of the symbol**                                   	|          **Example**                  |
|-------------	|-----------------------------------------------------------------------	| -----------------------------------	|
| `PROJECT_DIR`	| the home directory of the TOML configuration file (project directory) 	| `target = {{ PROJECT_DIR }}/model.py`     |

> **Note**: You can also use environmental variables. Just use `env` dict, e.g.: `{{ env['your_var_name'] }}`.

##### ✍️ Example
First, let's define some environmental variable: using Python or system tool.
```python
import os

os.environ["MY_LOG_LEVEL"] = "INFO"
```
or
```bash
export MY_LOG_LEVEL="MY_LOG_LEVEL"
```
Now, we can use the environmental variable `MY_LOG_LEVEL` in our config file:

```toml
[logging]
level = "{{ env['MY_LOG_LEVEL'] }}"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```
> **Warning**: If you use **double quote** for text values in TOML configuration file, then use **single quote** to access `env` values. 

#### Context constants
When you run training using `kit4dl train` command, all custom modules have access to context constant values (defined for the current Python interpreter session).
You can access them via `context` module:

##### ✍️ Example
```python
from kit4dl import context

print(context.PROJECT_DIR)
```

The constants currently available in `kit4dl` are the following:
|   **Symbol** 	|            **Meaning of the symbol**                                   	|          **Example**                  |
|-------------	|-----------------------------------------------------------------------	| -----------------------------------	|
| `PROJECT_DIR`	| the home directory of the TOML configuration file (project directory) 	| `context.PROJECT_DIR`                 |
| `LOG_LEVEL`	| logging level as defined in the configuration TOML file                	| `context.LOG_LEVEL`                   |
| `LOG_FORMAT`	| logging message format as defined in the configuration TOML file      	| `context.LOG_FORMAT`                  |
|  `VERSION`	| the current version of the package                                      	| `context.VERSION`                     |

