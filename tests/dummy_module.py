import torchmetrics as tm

import src.nn.confmodels
from src.dataset import AbstractDataset
from src.nn.base import AbstractModule


class A:
    pass


class B(AbstractModule):
    def __init__(self, input_dims, layers, dropout, output_dims):
        super().__init__()


class DummyDatasetModuleWrong:
    pass


class DummyDatasetModule(AbstractDataset):
    def __init__(self, train_config, root_dir: str):
        super().__init__(train_config, None, None)


class CustomMetricWrong:
    pass


class CustomMetric(tm.Metric):
    pass
