"""A module with predefined Kit4DL callbacks."""
import warnings

import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks

from kit4dl.nn.base import Kit4DLAbstractModule


class MetricCallback(pl_callbacks.Callback):
    """Callback that manages metrics.

    The callback resets metric stores on train or validation epoch start
    and logs metric on train or epoch end.
    """

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset train metric store on start of each training stage."""
        if not isinstance(pl_module, Kit4DLAbstractModule):
            warnings.warn(
                f"type `{type(pl_module)}` doesn't support MetricCallback"
                " logic. try to inherit from the `kit4dl.Kit4DLAbstractModule`"
            )
            return
        pl_module.train_metric_tracker.reset()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metric values on the end of training stage."""
        if not isinstance(pl_module, Kit4DLAbstractModule):
            warnings.warn(
                f"type `{type(pl_module)}` doesn't support MetricCallback"
                " logic. try to inherit from the `kit4dl.Kit4DLAbstractModule`"
            )
            return
        pl_module.log_train_metrics()

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset validation metric store on start of each validation stage."""
        if not isinstance(pl_module, Kit4DLAbstractModule):
            warnings.warn(
                f"type `{type(pl_module)}` doesn't support MetricCallback"
                " logic. try to inherit from the `kit4dl.Kit4DLAbstractModule`"
            )
            return
        pl_module.val_metric_tracker.reset()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metric values on the end of validation stage."""
        if not isinstance(pl_module, Kit4DLAbstractModule):
            warnings.warn(
                f"type `{type(pl_module)}` doesn't support MetricCallback"
                " logic. try to inherit from the `kit4dl.Kit4DLAbstractModule`"
            )
            return
        pl_module.log_val_metrics()

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset test metric store on start of each test stage."""
        if not isinstance(pl_module, Kit4DLAbstractModule):
            warnings.warn(
                f"type `{type(pl_module)}` doesn't support MetricCallback"
                " logic. try to inherit from the `kit4dl.Kit4DLAbstractModule`"
            )
            return
        pl_module.test_metric_tracker.reset()

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metric values on the end of test stage."""
        if not isinstance(pl_module, Kit4DLAbstractModule):
            warnings.warn(
                f"type `{type(pl_module)}` doesn't support MetricCallback"
                " logic. try to inherit from the `kit4dl.Kit4DLAbstractModule`"
            )
            return
        pl_module.log_test_metrics()


class ModelCheckpoint(pl_callbacks.ModelCheckpoint):
    """Callback for saving model checkpoint on fit end."""

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Save model weight on the end of training process."""
        super().on_fit_end(trainer, pl_module)
