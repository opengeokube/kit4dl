"""A module with neural network train task definition"""
import logging
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import callbacks
from lightning.pytorch import loggers as pl_log

from mlkit.dataset import MLKitAbstractDataset
from mlkit.mixins import LoggerMixin
from mlkit.nn.base import MLKitAbstractModule
from mlkit.nn.callbacks import MetricCallback, ModelCheckpoint
from mlkit.nn.confmodels import Conf
from mlkit.utils import set_seed


class Trainer(LoggerMixin):
    _model: MLKitAbstractModule
    _datamodule: MLKitAbstractDataset
    _trainer: pl.Trainer
    _conf: Conf
    _metric_logger: Any

    def __init__(self, conf: Conf) -> None:
        self._logger = logging.getLogger("lightning.pytorch")
        self._conf = conf
        self._device = self._conf.base.device
        self._metric_logger = self._new_metric_logger()
        set_seed(self._conf.base.seed)

    def prepare(self) -> "Trainer":
        self._model = self._configure_model()
        self._trainer = self._configure_trainer()
        self._datamodule = self._configure_datamodule()
        return self

    def fit(self) -> "Trainer":
        self._trainer.fit(self._model, datamodule=self._datamodule)
        return self

    def _new_metric_logger(self) -> pl_log.Logger:
        pass

    def _configure_datamodule(self) -> MLKitAbstractDataset:
        return self._conf.dataset.datamodule_class(conf=self._conf.dataset)

    def _configure_model(self) -> MLKitAbstractModule:
        return self._conf.model.model_class(conf=self._conf).to(self._device)

    def _get_model_checkpoint(self) -> ModelCheckpoint:
        chkp_conf = self._conf.training.checkpoint
        return ModelCheckpoint(
            dirpath=chkp_conf.path,
            filename=chkp_conf.filename,
            monitor=chkp_conf.monitor_metric,
            save_top_k=chkp_conf.save_top_k,
            mode=chkp_conf.mode,
            save_weights_only=chkp_conf.save_weights_only,
            every_n_train_steps=chkp_conf.every_n_train_steps,
            save_on_train_epoch_end=chkp_conf.save_on_train_epoch_end,
        )

    def _configure_trainer(self) -> pl.Trainer:
        accelerator_device, device = self._conf.base.accelerator_device_and_id
        callbacks = [MetricCallback()]
        if self._conf.training.checkpoint:
            callbacks.append(self._get_model_checkpoint())
        return pl.Trainer(
            accelerator=accelerator_device,
            devices=device,
            max_epochs=self._conf.training.epochs,
            check_val_every_n_epoch=self._conf.validation.run_every_epoch,
            enable_progress_bar=True,
            deterministic=True,
            logger=self._metric_logger,
            callbacks=callbacks,
        )
