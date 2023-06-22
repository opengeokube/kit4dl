"""A module with neural network train task definition"""
import logging
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_log

from mlkit.dataset import MLKitAbstractDataModule
from mlkit.mixins import LoggerMixin
from mlkit.nn.base import MLKitAbstractModule
from mlkit.nn.callbacks import MetricCallback, ModelCheckpoint
from mlkit.nn.confmodels import Conf
from mlkit.utils import set_seed


class Trainer(LoggerMixin):
    """Class managing the training procedure"""

    _model: MLKitAbstractModule
    _datamodule: MLKitAbstractDataModule
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
        """Prepare trainer by configuring the model and data modules"""
        self._model = self._configure_model()
        self._trainer = self._configure_trainer()
        self._datamodule = self._configure_datamodule()
        return self

    def fit(self) -> "Trainer":
        """Fit the trainer making use of `lightning.pytorch.Trainer`"""
        self._trainer.fit(self._model, datamodule=self._datamodule)
        return self

    def _new_metric_logger(self) -> pl_log.Logger:
        # TODO: prepare logger based on conf file: https://github.com/opengeokube/ml-kit/issues/2
        return pl_log.MLFlowLogger(experiment_name="test")

    def _configure_datamodule(self) -> MLKitAbstractDataModule:
        return self._conf.dataset.datamodule_class(conf=self._conf.dataset)

    def _configure_model(self) -> MLKitAbstractModule:
        return self._conf.model.model_class(conf=self._conf).to(self._device)

    def _get_model_checkpoint(self) -> ModelCheckpoint:
        assert self._conf.training.checkpoint, (
            "getting model checkpoint callback, but `checkpoint` was not"
            " defined in the configuration file"
        )
        chkp_conf = self._conf.training.checkpoint
        return ModelCheckpoint(
            dirpath=chkp_conf.path,
            filename=chkp_conf.filename,
            monitor=chkp_conf.monitor_metric_name,
            save_top_k=chkp_conf.save_top_k,
            mode=chkp_conf.mode,
            save_weights_only=chkp_conf.save_weights_only,
            every_n_train_steps=chkp_conf.every_n_train_steps,
            save_on_train_epoch_end=chkp_conf.save_on_train_epoch_end,
        )

    def _configure_trainer(self) -> pl.Trainer:
        accelerator_device, device = self._conf.base.accelerator_device_and_id
        callbacks: list[pl_callbacks.Callback] = [MetricCallback()]
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
