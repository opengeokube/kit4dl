# Example with handwritten digit classification

## To run the example, follow below instructions:

### 0. Install missing libraries
Install the following libraries:
1. `torchvision`
1. `matplotlib`

using PIP
```bash
pip install torchvision matplotlib
```
or `conda`.

### 1. Run
```bash
kit4dl train
```

while being in the directory with `conf.toml` file, or specify configuration file explicitly:

```bash
kit4dl train --conf=<path_to_conf_file>
```

> **Note**: The MNIST dataset will be downloaded automatically by `PyTorch`.