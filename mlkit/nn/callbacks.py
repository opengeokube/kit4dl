"""A module with predefined MLKit callbacks"""
import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks

from mlkit.nn.base import MLKitAbstractModule


class MetricCallback(pl_callbacks.Callback):
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: MLKitAbstractModule
    ) -> None:
        pl_module.train_metric_tracker.reset()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: MLKitAbstractModule
    ) -> None:
        pl_module.log_train_metrics()

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: MLKitAbstractModule
    ) -> None:
        pl_module.val_metric_tracker.reset()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: MLKitAbstractModule
    ) -> None:
        pl_module.log_val_metrics()

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: MLKitAbstractModule
    ) -> None:
        pl_module.test_metric_tracker.reset()

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: MLKitAbstractModule
    ) -> None:
        pl_module.log_test_metrics()
