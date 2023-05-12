"""A module with neural network train task definition"""
import lightning.pytorch as pl
from lightning.pytorch import callbacks
from lightning.pytorch import loggers as pl_log

from mlkit.dataset import MLKitAbstractDataset
from mlkit.nn.base import MLKitAbstractModule
from mlkit.nn.callbacks import MetricCallback, ModelCheckpoint
from mlkit.nn.confmodels import Conf
from mlkit.utils import set_seed


class Trainer:
    _model: MLKitAbstractModule
    _trainer: pl.Trainer
    _conf: Conf

    def __init__(self, conf: Conf) -> None:
        self._conf = conf
        self._device = self.conf.base.device

    def prepare(self) -> "Trainer":
        self._model = self._configure_model()
        self._trainer = self._configure_trainer()
        self._datamodule = self._configure_datamodule()
        return self

    def fit(self) -> "Trainer":
        self._trainer.fit(self._model, datamodule=self._datamodule)
        return self

    def _new_logger(self) -> pl_log.Logger:
        pass

    def _configure_datamodule(self) -> MLKitAbstractDataset:
        test_dataset_conf = (
            self._conf.test.dataset if self._conf.test is not None else None
        )
        predict_dataset_conf = (
            self._conf.predict.dataset if self._conf.test is not None else None
        )
        return self._conf.training.dataset.datamodule_class(
            train_config=self._conf.training.dataset,
            val_config=self._conf.validation.dataset,
            test_config=test_dataset_conf,
            predict_config=predict_dataset_conf,
        )

    def _configure_model(self) -> MLKitAbstractModule:
        return self.conf.model.model_class(**self.conf.model.arguments).to(
            self._device
        )

    def _configure_trainer(self) -> pl.Trainer:
        accelerator_device_id, device = self.conf.base.accelerator_device_id
        tconf = self.conf.training
        return pl.Trainer(
            accelerator_device_id=accelerator_device_id,
            devices=device,
            max_epochs=self.conf.training.epochs,
            check_val_every_n_epoch=self.conf.validation.run_every_epoch,
            enable_progress_bar=True,
            deterministic=True,
            logger=self._new_logger(),
            callbacks=[
                MetricCallback(),
                ModelCheckpoint(
                    dirpath=tconf.checkpoint.path,
                    filename=tconf.checkpoint.filename,
                    monitor=tconf.checkpoint.monitor,
                    save_top_k=tconf.checkpoint.save_top_k,
                    mode=tconf.checkpoint.mode,
                    save_weights_only=tconf.checkpoint.save_weights_only,
                    every_n_train_steps=tconf.checkpoint.every_n_train_steps,
                    save_on_train_epoch_end=tconf.checkpoint.save_on_train_epoch_end,
                ),
            ],
        )
