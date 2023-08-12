import torchmetrics as tm

from kit4dl.dataset import MLKitAbstractDataModule
from kit4dl.nn.base import MLKitAbstractModule


class A:
    pass


class B(MLKitAbstractModule):
    def setup(self, input_dims, layers, dropout, output_dims):
        pass

    def run_step(self, batch, batch_idx):
        pass


class DummyDatasetModuleWrong:
    pass


class DummyDatasetModule(MLKitAbstractDataModule):
    def __init__(self, train_config, root_dir: str):
        super().__init__(train_config, None, None)


class CustomMetricWrong:
    pass


class CustomMetric(tm.Metric):
    pass


class T1:
    def f1():
        pass
