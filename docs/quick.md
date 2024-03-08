# Quick start
Below, you can find quick start quide for **Kit4DL**.

## Step 1. Create a virtual environment
You can use `venv` built-it package:

``` { .bash .copy }
python -m venv ai
```
and activate it using appropriate file in `ai/bin` directory.

You can also use conda environments:


``` { .bash .copy }
conda create -n ai python=3.11
```
and activate it using the command


``` { .bash .copy }
conda activate ai
```


## Step 2. Install Kit4DL
Either using `PyPI` repository:

``` { .bash .copy }
pip install kit4dl 
```

or using Anaconda repository:

``` { .bash .copy }
conda install -c conda-forge kit4dl
```

## Step 3. Prepare an empty Kit4DL project:

``` { .python .annotate }
kit4dl init --name=my-first-project # (1)!
```

1.  Parameter `--name` is optional and, by default, takes value `new_kit4dl_project`


## Step 4. Prepare TOML configuration file
All configuration stuff related to ANN learning, validation and testing is managed by the configuration file in TOML format.
When you create an empty Kit4DL project, you'll get example config.toml file with the following structure:

``` { .toml .annotate }
[base] # (1)!
seed = 0 # (2)!
cuda_id = 0 # (3)!
experiment_name = "my_new_awesome_experiment" # (4)!

[logging] # (5)!
type = "comet" # (6)!
level = "info" # (7)!
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" # (8)!


[model] # (9)!
target = "./model.py::MyNewNetwork" # (10)!
arg1 = "..." # (11)!
arg2 = "..."

[training] # (12)!
epochs = 100 # (13)!
epoch_schedulers = [ # (14)!
    {target = "torch.optim.lr_scheduler::CosineAnnealingLR", T_max=100}
]

[training.checkpoint] # (15)!
path = "my_checkpoints" # (16)!
monitor = {metric = "MyScore", stage = "val"} # (17)!
filename = "{epoch}_{val_myscore:.2f}_some_name" # (18)!
mode = "max" # (19)!
save_top_k = 1

[training.optimizer]
target = "torch.optim::Adam" # (20)!
lr = 0.0001 # (21)!
weight_decay = 0.03

[training.criterion]
target = "torch.nn::NLLLoss" # (22)!
weight = [0.1, 0.9] # (23)!

[validation]
run_every_epoch = 1 # (24)!

[dataset]
target = "./datamodule.py::MyCustomDatamodule" # (25)!

[dataset.trainval] # (26)!
arg1 = 10 # (27)!
arg2 = ...

[dataset.train] # (28)!
root_dir = "my_train_file.txt" # (29)!

[dataset.validation]
root_dir = "my_val_file.txt"

[dataset.train.loader] # (30)!
batch_size = 10 # (31)!
shuffle = true
num_workers = 4

[dataset.validation.loader] # (32)!
batch_size = 10
shuffle = false
num_workers = 4

[dataset.test.loader] # (33)!
batch_size = 10
shuffle = false
num_workers = 4

[metrics] # (34)!
MyScore = { 
    target = "torchmetrics::Precision",  # (35)!
    task = "multiclass", # (36)!
    num_classes = 10}  
MySecondScore = {target = "torchmetrics::FBetaScore", task = "multiclass", num_classes = 10, beta = 0.1} 
```

1. The section with base setup.
2. Seed for pseudorandom numbers generator (to enable reproducibility).
3. CUDA device ID to use (if you have just one GPU installed, use `0`). To use CPU for learning, remove that line.
4. Experiment name to use throughout the learning/logging process. Basically, it can be any text you want.
5. Section for logging setup.
6. One of supported logging services to use ("comet", "csv", "mlflow", "neptune", "tensorboard", "wandb"). If not set, `csv` will be used.
7. Logging level for Python logger ("debug", "info", "warn", "error").
8. Logging format according to Python logging documentation. Use [supported attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes).
9. ANN-model related configuration.
10. Path to the subclass of `Kit4DLAbstractModule`. If can be an absolute or relative path followed by class name (e.g. `./model.py::MyNewNetwork`). If the class is importable, you can specify its fully qualified name (e.g. `some_package.some_module.MyNewNetwork`).
11. If you class `MyNewNetwork` have some parameters, you can specify them here (e.g. number of layers, number of hidden units, etc).
12. The section with trening-specific configuration.
13. Quite self-explanatory: number of epochs :)
14. If you want to specify some epochs schedulers, you can add `epoch_schedulers` attribute in the configuration file. It should be a list of dictionaries. Each dictionary should have `target` attribute indicating a class to the scheduler, and - if needed - other arguments required by the scheduler class indicated in `target` property.
15. You can specify model checkpointing!
16. Directory where checkpoints should be saved.
17. Attribute to indicate which metric should be tracked for checkpointing (e.g. to save the model with the best accuracy) and for which phase (suuported phases are: `train`, `val`).
18. Pattern for saved checkpoint as indicated [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html). You can use metric defined in `[metric]` section using the pattern `<stage>_<lowercase metric name>` (e.g. `val_myscore`).
19. You can also define some other arguments accepted by [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
20. Path to the optimizer class.
21. The optimizer-specify arguments can be specified as extra attributes in the section.
22. Path to the criterion class
23. Weights for criterion specified as list of floats.
24. How frequently validation should be run.
25. Path to the subclass of `Kit4DLAbstractDataModule`
26. If there is a common procedure for generating train/validation sSlit like random split the input data, you can define arguments in the section [dataset.trainval]. 
27. Arguments for the `prepare_trainvaldataset` method of you subclass of `Kit4DLAbstractDataModule` might be passed as extra attributes in this section.
28. If you want to specify different logic for train and validation datasets like loading separate formerly prepared files, you can use separate sections [dataset.train] and [dataset.validation].
29. This is a sample argument passed to `prepare_traindataset` of you subclass of `Kit4DLAbstractDataModule`.
30. Section for configuration of data loader for train dataset.
31. You can define all arguments accepted by [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
32. Similarily for validation data loader...
33. ... and the test one.
34. The section contains metric used for evaluate model, save checkpoints, and store in services like [Comet.ml](http://comet.com). Each metric is a dictionary whose key must be an only-letter text without white signs...
35. `target` key for a metric should indicate a subclass of `torchmetric.Metric` base class or one of the metrics defined in `torchmetric` package (e.g. `torchmetrics::Precision`).
36. Some extra arguments required by the given metric can be specified as other items in the dictionary.


