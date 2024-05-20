# Command Line Interface (CLI)

Kit4DL provides users with a conveninent CLI to create, run, or resume experiments.

## Creating an empty project
In order to create an empty project in the current working directory, use the following command:

``` { .bash .copy }
kit4dl init
```

If you want to specify a custom name for the experiment directory, use `--name` option:

``` { .bash .copy }
kit4dl init --name=my-custom-project
```


## Running train-validation loop
To run the training just type the following command:

``` { .bash .copy }
kit4dl train
```

If you want to run also test for best saved weight, use flag `--test`:

``` { .bash .copy }
kit4dl train --test
```

## Specifying path to the configuration file

``` { .bash .copy }
kit4dl train --conf=/path/to/your/conf.toml
```

> **Note**: By default, Kit4DL searches for the `conf.toml` file in the current working directory. If you want to specify the other path or the name of your configuration file differs from the expected one (`conf.toml`), use `--config` option:

## Overwriting configuration
!!! Note
    Available since 2024.5b0

You can overwrite any configuration option with CLI argument called `overwrite`. 
Just specify a comma-separated string of options to replace. Nested keys are available using `.` (dot operator).
Below, you can see an example.

To overwrite logging level to `ERROR` and to change learning rate to `0.5`, just run

``` { .bash .copy }
kit4dl train --conf=/path/to/your/conf.toml --overwrite "logging.level=error,training.optimizer.lr=0.5"
```