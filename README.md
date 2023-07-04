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

> **Warning**: MLKit is currently in its alpha stage. All recommendations are welcomed.

- [ ] add handling sklearn-like models
- [ ] add functionality to serve the model
- [ ] enable custom metrics
- [ ] write more unit tests


## 🎬 Quickstart

### Getting started

#### Installation
```bash
pip install mlkit
```

or

```bash
conda install -c conda-forge mlkit
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

## 🪁 Playground
At first, install `mlkit` package as indicated in the Section [Installation](#installation).

#### Handwritten digit recognition
Just navigate to the directory `/examples/cnn_mnist_classification` and run
```bash
mlkit train
```

#### Point cloud instance segmentation
Just navigate to the directory `/examples/cnn_s3dis_segmentation` and run
```bash
mlkit train
```


## 💡 Instruction
1. [Configuring base setup](#configuring-base-setup)
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
|   `log_level`     |  `str`           | logging level, should be one out of `debug`, `info`, `warn`, `error`, `critical`                         |
|  `log_format`     | `str`            | format of the logging message accoridng to Python's `logging` package  (e.g. `"%(asctime)s - %(name)s`)  |

> **Note**: Arguments marked with `*` are obligatory!


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
    def configure(self, input_dims, output_dims) -> None:
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
> **Note**: `run_step` method should return a tuple of two tensors, the ground-truth labels and the output of the network.

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
> **Note**: `target` is a required parameter that **must** be set. It contains a path to the class (a subclass of `MLKitAbstractModule`). To learn how `target` could be defined, see Section [Defining `target`](#defining-target).

If a forward pass for your model differs for the training, validation, test, or prediction stages, you can define separate methods for them:

##### ✍️ Example
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

> **Note**: If you need more customization of the process, you can always override the existing methods according to your needs.

#### Defining datamodule
Similarily to the model, datamodule instance is fully defined by the Python class and its configuration.
The datamodule need to be a subclass of the `MLKitAbstractDataModule` abstract class from the `mlkit` package.
The class has to implement, at least, `prepare_trainvaldataset` (if preparing is the same for the train and validation splits) or `prepare_traindataset` and `prepare_valdataset` (if preparing data differs). Besides those, you can define `prepare_testdataset` and `prepare_predictdataset`, for test and prediction, respectively.

##### ✍️ Example
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

##### ✍️ Example
```python
...
class MNISTCustomDatamodule(MLKitAbstractDataModule):
    ...
    def prepare_data(self):
        ...
```

> **Warning**: **DO NOT** set state inside `prepare_data` method (~~`self.x = ...`~~).

If you need more customization, feel free to override the other methods of `MLKitAbstractDataModule` superclass.
To force custom batch collation, override selected methods out of the following ones. They should return the proper callable object!

```python
def some_collate_func(samples: list): ...

class MNISTCustomDatamodule(MLKitAbstractDataModule):
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

In the root `[dataset]` you should define `target` property being a path to the subclass of the `MLKitAbstractDataModule` module (see [Defining `target`](#defining-target)).
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
|`epoch_schedulers` |  `list of dict`  |  list of schedulers definitions  |

> **Note**: Arguments marked with `*` are obligatory!

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
path = "${PROJECT_DIR}/chckpt"
monitor = {"metric" = "Precision", "stage" = "val"}
filename = "{epoch}_{val_precision:.2f}_cnn"
mode = "max"
save_top_k = 1
```

> **Note**: You can see we used substitutable symbol `${PROJECT_DIR}`. More about them in the Section [Substitutable symbols](#substitutable-symbols).




#### Defining `target`
Target property in the MLKit package is kind of extended fully qualified name pointing to the classes supposed to use in the
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
The symbols should be used in curly brackets (e.g. `${PROJECT_DIR}`.)

|   **Symbol** 	|            **Meaning of the symbol**                                   	|          **Example**                  |
|-------------	|-----------------------------------------------------------------------	| -----------------------------------	|
| `PROJECT_DIR`	| the home directory of the TOML configuration file (project directory) 	| `target = ${PROJECT_DIR}/model.py`     |



#### Context constants
When you run training using `mlkit train` command, all custom modules have access to context constant values (defined for the current Python interpreter session).
You can access them via `context` module:

##### ✍️ Example
```python
from mlkit import context

print(context.PROJECT_DIR)
```

The constants currently available in `mlkit` are the following:
|   **Symbol** 	|            **Meaning of the symbol**                                   	|          **Example**                  |
|-------------	|-----------------------------------------------------------------------	| -----------------------------------	|
| `PROJECT_DIR`	| the home directory of the TOML configuration file (project directory) 	| `context.PROJECT_DIR`                 |
| `LOG_LEVEL`	| logging level as defined in the configuration TOML file                	| `context.LOG_LEVEL`                   |
| `LOG_FORMAT`	| logging message format as defined in the configuration TOML file      	| `context.LOG_FORMAT`                  |

