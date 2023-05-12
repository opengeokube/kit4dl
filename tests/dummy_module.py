import torchmetrics as tm

from mlkit.dataset import MLKitAbstractDataset
from mlkit.nn.base import MLKitAbstractModule


class A:
    pass


class B(MLKitAbstractModule):
    def setup(self, input_dims, layers, dropout, output_dims):
        pass

    def step(self, batch, batch_idx):
        pass


class DummyDatasetModuleWrong:
    pass


class DummyDatasetModule(MLKitAbstractDataset):
    def __init__(self, train_config, root_dir: str):
        super().__init__(train_config, None, None)


class CustomMetricWrong:
    pass


class CustomMetric(tm.Metric):
    pass


class T1:
    def f1():
        pass
