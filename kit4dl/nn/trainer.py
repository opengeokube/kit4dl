"""A module with neural network train task definition."""

import os

import torch
import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_log

from kit4dl.nn.dataset import Kit4DLAbstractDataModule
from kit4dl.mixins import LoggerMixin
from kit4dl.nn.base import Kit4DLModuleWrapper
from kit4dl.nn.callbacks import MetricCallback
from kit4dl.nn.confmodels import Conf
from kit4dl.utils import set_seed
from kit4dl import io as io_


class Trainer(LoggerMixin):
    """Class managing the training procedure."""

    _conf: Conf
    _device: torch.device
    _metric_logger: pl_log.Logger
    model_wrap: Kit4DLModuleWrapper
    _datamodule: Kit4DLAbstractDataModule
    pl_trainer: pl.Trainer

    def __init__(self, conf: Conf) -> None:
        self._conf = conf
        self._configure_logger()
        self._device = self._conf.base.device
        self._metric_logger = self._new_metric_logger()
        set_seed(self._conf.base.seed)

    def prepapre_new(self, *, experiment_name: str | None = None) -> "Trainer":
        """Copy and prepare the trainer.

        Parameters
        ----------
        experiment_name : str, optional
            Name of the experiment to be used in the new trainer. If not
            provided, the name is copied from the current trainer.
        """
        _conf = self._conf
        if experiment_name:
            _conf = self._conf.copy()
            _conf.base.experiment_name = experiment_name
            _conf.logging.maybe_update_experiment_name(experiment_name)
        return type(self)(_conf).prepare()

    def _configure_logger(self) -> None:
        super().configure_logger(
            name="lightning.pytorch",
            level=self._conf.logging.level,
            logformat=self._conf.logging.format_,
        )

    @property
    def is_finished(self) -> bool:
        """Check if training routing finished."""
        return self.pl_trainer.state.finished

    def prepare(self) -> "Trainer":
        """Prepare trainer by configuring the model and data modules."""
        self.model_wrap = self._wrap_model()
        self.pl_trainer = self._configure_trainer()
        self._datamodule = self._configure_datamodule()
        self._log_hparams()
        return self

    def _log_hparams(self) -> None:
        self._metric_logger.log_hyperparams(self._conf.obfuscated_dict())
        self._metric_logger.log_hyperparams(
            {
                "trainable_parameters": sum(
                    p.numel()
                    for p in self.model_wrap.model.parameters()
                    if p.requires_grad
                )
            }
        )

    def fit(self) -> "Trainer":
        """Fit the trainer making use of `lightning.pytorch.Trainer`."""
        assert self.pl_trainer, (
            "trainer is not configured. did you forget to call `prepare()`"
            " method first?"
        )
        self._datamodule.setup("fit")
        new_trainer = self
        i = 0
        for i, (tr_dataloader, val_dataloader, suff) in enumerate(
            self._datamodule.trainval_dataloaders()
        ):
            self._logger.info("Starting training for split %d...", i + 1)
            new_trainer.pl_trainer.fit(
                new_trainer.model_wrap.model,
                train_dataloaders=tr_dataloader,
                val_dataloaders=val_dataloader,
            )
            new_trainer = new_trainer.prepapre_new(
                experiment_name=self._conf.experiment_name + suff
            )
        self._logger.info("Training finished! %d splits processed", i + 1)
        return self

    def test(self) -> "Trainer":
        """Test the model."""
        assert self.pl_trainer, (
            "trainer is not configured. did you forget to call `prepare()`"
            " method first?"
        )
        ckpt_path = None
        for callback in self.pl_trainer.checkpoint_callbacks:
            if isinstance(callback, pl_callbacks.ModelCheckpoint):
                self.debug(
                    "best checkpoint taken from callback %s",
                    callback.best_model_path,
                )
                ckpt_path = callback.best_model_path
                break
        if self._conf.training.checkpoint_path:
            assert os.path.exists(self._conf.training.checkpoint_path), (
                "the defined checkpoint:"
                f" {self._conf.training.checkpoint_path} does not exist!"
            )
            self.info(
                "user-defined checkpoint %s will be used for testing",
                self._conf.training.checkpoint_path,
            )
            ckpt_path = self._conf.training.checkpoint_path
        if ckpt_path:
            model = self.model_wrap.load_checkpoint(ckpt_path)
        self._datamodule.setup("test")
        for i, test_dataloader in enumerate(
            self._datamodule.test_dataloader()
        ):
            self._logger.info("Starting testing for split %d...", i + 1)
            self.pl_trainer.test(model, dataloaders=test_dataloader)
        return self

    def predict(self) -> "Trainer":
        """Predict values for the model."""
        assert self.pl_trainer, (
            "trainer is not configured. did you forget to call `prepare()`"
            " method first?"
        )
        self._datamodule.setup("predict")
        for i, pred_dataloader in enumerate(
            self._datamodule.predict_dataloader()
        ):
            self._logger.info("Starting prediction for split %d...", i + 1)
            self.pl_trainer.predict(
                self.model_wrap.model, dataloaders=pred_dataloader
            )
        return self

    def _new_metric_logger(self) -> pl_log.Logger:
        return self._conf.logging.metric_logger_type(
            **self._conf.logging.arguments
        )

    def _configure_datamodule(self) -> Kit4DLAbstractDataModule:
        class_ = self._conf.dataset.datamodule_class
        io_.assert_valid_class(class_, Kit4DLAbstractDataModule)
        return self._conf.dataset.datamodule_class(conf=self._conf.dataset)

    def _wrap_model(self) -> Kit4DLModuleWrapper:
        return Kit4DLModuleWrapper(conf=self._conf, device=self._device)

    def _get_model_checkpoint(self) -> pl_callbacks.ModelCheckpoint:
        assert self._conf.training.checkpoint, (
            "getting model checkpoint callback, but `checkpoint` was not"
            " defined in the configuration file"
        )
        chkp_conf = self._conf.training.checkpoint
        assert (not chkp_conf.every_n_epochs) or isinstance(
            chkp_conf.every_n_epochs, int
        ), (
            "wrong type of `every_n_epochs`. expected: `int`, provided:"
            f" {type(chkp_conf.every_n_epochs)}"
        )
        return pl_callbacks.ModelCheckpoint(
            dirpath=chkp_conf.path,
            filename=chkp_conf.filename,
            monitor=chkp_conf.monitor_metric_name,
            save_top_k=chkp_conf.save_top_k,
            mode=chkp_conf.mode,
            save_weights_only=chkp_conf.save_weights_only,
            every_n_epochs=chkp_conf.every_n_epochs,
            save_on_train_epoch_end=chkp_conf.save_on_train_epoch_end,
        )

    def _set_default_trainer_args(self):
        self._conf.training.arguments.setdefault("deterministic", True)
        self._conf.training.arguments.setdefault("enable_progress_bar", True)

    def _configure_trainer(self) -> pl.Trainer:
        accelerator_device, device = self._conf.base.accelerator_device_and_id
        callbacks: list[pl_callbacks.Callback] = [
            MetricCallback(conf=self._conf.metrics_obj)
        ] + self._conf.training.preconfigured_callbacks
        if self._conf.training.checkpoint:
            callbacks.append(self._get_model_checkpoint())
        return pl.Trainer(
            accelerator=accelerator_device,
            devices=device,
            max_epochs=self._conf.training.epochs,
            check_val_every_n_epoch=self._conf.validation.run_every_epoch,
            logger=self._metric_logger,
            callbacks=callbacks,
            **self._conf.training.arguments,
        )
