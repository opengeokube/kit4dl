import torchmetrics as tm

from kit4dl.dataset import Kit4DLAbstractDataModule
from kit4dl.nn.base import Kit4DLAbstractModule


class A:
    pass


class B(Kit4DLAbstractModule):
    def setup(self, input_dims, layers, dropout, output_dims):
        pass

    def run_step(self, batch, batch_idx):
        pass


class DummyDatasetModuleWrong:
    pass


class DummyDatasetModule(Kit4DLAbstractDataModule):
    def __init__(self, train_config, root_dir: str):
        super().__init__(train_config, None, None)


class CustomMetricWrong:
    pass


class CustomMetric(tm.Metric):
    pass


class T1:
    def f1():
        pass
