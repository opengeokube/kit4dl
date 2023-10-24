"""A module with predefined Kit4DL callbacks."""

from typing import Any
import logging

import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch.utilities.types import STEP_OUTPUT

from kit4dl import context
from kit4dl.metric import MetricStore
from kit4dl.stages import Stage
from kit4dl.mixins import LoggerMixin


class MetricCallback(pl_callbacks.Callback, LoggerMixin):
    """Callback that manages metrics.

    The callback resets metric stores on train or validation epoch start
    and logs metric on train or epoch end.
    """

    _logger: logging.Logger

    def __init__(self, conf: dict) -> None:
        super().__init__()
        super().configure_logger(
            name="kit4dl.MetricCallback",
            level=context.LOG_LEVEL,
            logformat=context.LOG_FORMAT,
        )
        self.train_metric_tracker = MetricStore(conf)
        self.val_metric_tracker = MetricStore(conf)
        self.test_metric_tracker = MetricStore(conf)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset train metric store on start of each training stage."""
        self.train_metric_tracker.reset()

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        """Accumulate batch results."""
        assert isinstance(outputs, dict), "output of the step is not a dict"
        self.train_metric_tracker.update(
            true=outputs["true"], predictions=outputs["pred"]
        )
        pl_module.log(
            name=f"{Stage.TRAIN}_loss", value=outputs["loss"], logger=True
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metric values on the end of training stage."""
        for (
            metric_name,
            metric_value,
        ) in self.train_metric_tracker.results.items():
            stage_metric_name = f"{Stage.TRAIN}_{metric_name}"
            self.info(
                "epoch: %d metric: %s value: %s",
                pl_module.current_epoch,
                stage_metric_name,
                metric_value,
            )
            pl_module.log(
                stage_metric_name,
                metric_value,
                logger=True,
            )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset validation metric store on start of each validation stage."""
        self.val_metric_tracker.reset()

    def on_validation_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Accumulate batch results."""
        assert isinstance(outputs, dict), "output of the step is not a dict"
        self.train_metric_tracker.update(
            true=outputs["true"], predictions=outputs["pred"]
        )
        pl_module.log(
            name=f"{Stage.VALIDATION}_loss", value=outputs["loss"], logger=True
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metric values on the end of validation stage."""
        for (
            metric_name,
            metric_value,
        ) in self.val_metric_tracker.results.items():
            stage_metric_name = f"{Stage.VALIDATION}_{metric_name}"
            self.info(
                "epoch: %d metric: %s value: %s",
                pl_module.current_epoch,
                stage_metric_name,
                metric_value,
            )
            pl_module.log(
                stage_metric_name,
                metric_value,
                logger=True,
            )

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset test metric store on start of each test stage."""
        self.test_metric_tracker.reset()

    def on_test_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Accumulate batch results."""
        assert isinstance(outputs, dict), "output of the step is not a dict"
        self.train_metric_tracker.update(
            true=outputs["true"], predictions=outputs["pred"]
        )
        pl_module.log(
            name=f"{Stage.TEST}_loss", value=outputs["loss"], logger=True
        )

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metric values on the end of test stage."""
        for (
            metric_name,
            metric_value,
        ) in self.test_metric_tracker.results.items():
            stage_metric_name = f"{Stage.TEST}_{metric_name}"
            self.info(
                "epoch: %d metric: %s value: %s",
                pl_module.current_epoch,
                stage_metric_name,
                metric_value,
            )
            pl_module.log(
                stage_metric_name,
                metric_value,
                logger=True,
            )


class ModelCheckpoint(pl_callbacks.ModelCheckpoint):
    """Callback for saving model checkpoint on fit end."""

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Save model weight on the end of training process."""
        super().on_fit_end(trainer, pl_module)
