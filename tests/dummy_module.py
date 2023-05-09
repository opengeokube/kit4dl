import torchmetrics as tm

from src.dataset import AbstractDataset
from src.nn.base import MLKitAbstractModule


class A:
    pass


class B(MLKitAbstractModule):
    def __init__(self, input_dims, layers, dropout, output_dims):
        super().__init__(conf=None)

    def step(self, batch, batch_idx):
        pass


class DummyDatasetModuleWrong:
    pass


class DummyDatasetModule(AbstractDataset):
    def __init__(self, train_config, root_dir: str):
        super().__init__(train_config, None, None)


class CustomMetricWrong:
    pass


class CustomMetric(tm.Metric):
    pass
