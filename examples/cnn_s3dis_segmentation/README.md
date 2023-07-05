# Example with point cloud semantic segmentation

## To run the example, follow below instructions:


### 1. Download S3DIS dataset
Follow the instructions in (http://buildingparser.stanford.edu/dataset.html)[http://buildingparser.stanford.edu/dataset.html]

### 2. Data location

#### 2a. Either get directories `Area_1` - `Area_6` in the `data` catalog on the example.
#### 2b. Modify the `s3dis.py` to consider the directory of the S3DIS dataset (mainly line 70.)

### 3. Run
```bash
mlkit train
```

while being in the directory with `conf.toml` file, or specify configuration file explicitly:

```bash
mlkit train --conf=<path_to_conf_file>
```
