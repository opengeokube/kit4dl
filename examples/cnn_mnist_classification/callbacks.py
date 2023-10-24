"""Extra callbacks definitions."""

import os
from typing import Any

import lightning.pytorch as pl
import torchmetrics as tm

from kit4dl import Kit4DLCallback, StepOutput


class SaveConfusionMatrixCallback(Kit4DLCallback):
    _cm: tm.ConfusionMatrix
    _num_classes: int
    _task: str
    _save_dir: str

    def __init__(self, task: str, num_classes: int, save_dir: str) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._save_dir = save_dir
        self._task = task
        os.makedirs(self._save_dir, exist_ok=True)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._cm = tm.ConfusionMatrix(
            task=self._task, num_classes=self._num_classes
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: StepOutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._cm.update(outputs.predictions, outputs.labels)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called when the val epoch ends."""
        fig, _ = self._cm.plot()
        target_file = os.path.join(
            self._save_dir,
            f"confusion_matrix_for_epoch_{pl_module.current_epoch}",
        )
        fig.savefig(target_file)
