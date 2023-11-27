"""A module with the base class of modules supported by Kit4DL."""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import lightning.pytorch as pl
import torch

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import kit4dl.io as io_
from kit4dl.mixins import LoggerMixin
from kit4dl.nn.confmodels import Conf


class _Criterion:
    def __init__(
        self,
        criterion: torch.nn.Module | Callable | None = None,
        *,
        conf: Conf,
    ):
        self._criterion = criterion
        if isinstance(self._criterion, torch.nn.Module):
            self._criterion = self._criterion.to(conf.base.device)

    @property
    def is_defined(self) -> bool:
        """Check if criterion is defined."""
        return self._criterion is not None

    def __call__(self, *ar, **kw):
        return self._criterion(*ar, **kw)

    def __repr__(self) -> str:
        return repr(self._criterion)


class Kit4DLModuleWrapper(LoggerMixin):
    """Kit4DL wrapper class around Kit4DLAbstractModule."""

    _conf: Conf
    model: Kit4DLAbstractModule

    def __init__(
        self, conf: Conf, *, device: torch.device | None = None
    ) -> None:
        super().__init__()
        assert conf, "`conf` argument cannot be `None`"
        self._conf: Conf = conf
        self._configure_logger()
        self.model = self._prepare_model()
        if device:
            self.debug("moving model to %s", device)
            self.model = self.model.to(device)

    def load_checkpoint(self, path: str) -> Kit4DLAbstractModule:
        """Load model weights from the checkpoint."""
        model = type(self.model).load_from_checkpoint(path)
        model.freeze()
        return self._configure_model(model)

    def _prepare_model(self) -> Kit4DLAbstractModule:
        io_.assert_valid_class(
            self._conf.model.model_class, Kit4DLAbstractModule
        )
        model = self._conf.model.model_class(**self._conf.model.arguments)
        return self._configure_model(model)

    def _configure_model(self, model) -> Kit4DLAbstractModule:
        model.criterion = self._configure_criterion()
        model.configure_logger(
            name="lightning",
            level=self._conf.logging.level,
            logformat=self._conf.logging.format_,
        )
        model.configure_optimizers = self.configure_optimizers
        return model

    def _configure_criterion(self) -> _Criterion:
        if not self._conf.training.criterion:
            self.info(
                "criterion was not set! remember to return loss value in the"
                " proper run methods!"
            )
            return _Criterion(conf=self._conf)
        self.debug("configuring criterion...")
        crt = self._conf.training.criterion.criterion
        self.info("selected criterion is: %s", crt)
        return _Criterion(crt, conf=self._conf)

    def _configure_logger(self) -> None:
        super().configure_logger(
            name="lightning",
            level=self._conf.logging.level,
            logformat=self._conf.logging.format_,
        )

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[LRScheduler] | None,
    ]:
        """Configure optimizers and schedulers."""
        self.debug("configuring optimizers and lr epoch schedulers...")
        optimizer: torch.optim.Optimizer = (
            self._conf.training.optimizer.optimizer(self.model.parameters())
        )
        lr_schedulers: list = [
            scheduler(optimizer)
            for scheduler in self._conf.training.preconfigured_schedulers_classes
        ]
        self.info("selected optimizer is: %s", optimizer)
        self.info(
            "selected %d  lr schedulers: %s", len(lr_schedulers), lr_schedulers
        )
        return [optimizer], lr_schedulers


class Kit4DLAbstractModule(
    ABC, pl.LightningModule, LoggerMixin
):  # pylint: disable=too-many-ancestors
    """Base abstract class for Kit4DL modules."""

    criterion: _Criterion

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.configure(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def _kit4dl_logger(self) -> logging.Logger:
        return self._logger

    @abstractmethod
    def configure(self, *args: Any, **kwargs: Any) -> None:
        """Configure the architecture of the neural network.

        Parameters
        ----------
        *args: Any
            List of positional arguments to setup the network architecture
        **kwargs : Any
            List of named arguments required to setup the network architecture

        Examples
        --------
        ```python
        def configure(self, input_dims, output_dims) -> None:
            self.fc1 = nn.Sequential(
                nn.Linear(input_dims, output_dims),
            )
        ```
        """
        raise NotImplementedError

    @abstractmethod
    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
        """Carry out single train/validation/test step for the given `batch`.

        Return a tuple of two `torch.Tensor`'s: true labels and predicted scores.
        If you need to define separate logic for validation or test step,
        implement `val_step` or `test_step` methods, respectivelly.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch

        Returns
        -------
        result : tuple of `torch.Tensor`
            A tuple of 2 or 3 items:
            - if a tuple of 2 elements:
                1. `torch.Tensor` of ground-truth labels,
                2. `torch.Tensor` output of the network,
            - if a tuple of 3 elements:
                1. `torch.Tensor` of ground-truth labels,
                2. `torch.Tensor` output of the network,
                4. `torch.Tensor` with loss value.

        Examples
        --------
        ```python
        ...
        def run_step(self, batch, batch_idx) -> tuple[torch.Tensor,  ...]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```

        ```python
        ...
        def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
            feature_input, label_input = batch
            scores = self(feature_input)
            loss = super().compute_loss(prediction=logits, target=is_fire)
            return (label_input, scores, loss)
        ```
        """
        raise NotImplementedError

    def run_val_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
        """Carry out single validation step for the given `batch`.

        Return a tuple of two `torch.Tensor`'s: true labels and predicted scores.
        If you need to define separate logic for validation or test step,
        implement `val_step` or `test_step` methods, respectivelly.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch

        Returns
        -------
        result : tuple of `torch.Tensor`
            A tuple of 2 or 3 items:
            - if a tuple of 2 elements:
                1. `torch.Tensor` of ground-truth labels,
                2. `torch.Tensor` output of the network,
            - if a tuple of 3 elements:
                1. `torch.Tensor` of ground-truth labels,
                2. `torch.Tensor` output of the network,
                4. `torch.Tensor` with loss value.

        Examples
        --------
        ```python
        ...
        def run_val_step(self, batch, batch_idx) -> tuple[torch.Tensor,  ...]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```

        ```python
        ...
        def run_val_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
            feature_input, label_input = batch
            scores = self(feature_input)
            loss = super().compute_loss(prediction=logits, target=is_fire)
            return (label_input, scores, loss)
        ```
        """
        return self.run_step(batch, batch_idx)

    def run_test_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
        """Carry out single test step for the given `batch`.

        Return a tuple of two `torch.Tensor`'s: true labels and predicted scores.
        If you need to define separate logic for validation or test step,
        implement `val_step` or `test_step` methods, respectivelly.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch

        Returns
        -------
        result : tuple of `torch.Tensor`
            A tuple of 2 or 3items:
            - if a tuple of 2 elements:
                1. `torch.Tensor` of ground-truth labels,
                2. `torch.Tensor` output of the network,
            - if a tuple of 3 elements:
                1. `torch.Tensor` of ground-truth labels,
                2. `torch.Tensor` output of the network,
                4. `torch.Tensor` with loss value.

        Examples
        --------
        ```python
        ...
        def run_test_step(self, batch, batch_idx) -> tuple[torch.Tensor,  ...]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```

        ```python
        ...
        def run_test_step(self, batch, batch_idx) -> tuple[torch.Tensor, ...]:
            feature_input, label_input = batch
            scores = self(feature_input)
            loss = super().compute_loss(prediction=logits, target=is_fire)
            return (label_input, scores, loss)
        ```
        """
        return self.run_step(batch, batch_idx)

    def run_predict_step(self, batch, batch_idx) -> torch.Tensor:
        """Carry out single predict step for the given `batch`.

        Return a `torch.Tensor` - the predicted scores.
        If not overriden, the implementation of `step` method is used.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch

        Returns
        -------
        result : torch.Tensor
            The score being the output of of the network


        Note
        ----
        The function returns just score values as for prediction we do not have
        the ground-truth labels.

        Examples
        --------
        ```python
        ...
        def run_predict_step(self, batch, batch_idx) -> torch.Tensor:
            feature_input = batch
            return self(feature_input)
        ```
        """
        _, scores = self.run_step(batch, batch_idx)
        return scores

    def _prepare_step_output(self, *, pred, true, loss, **kw) -> dict:
        return {"loss": loss, "pred": pred, "true": true} | kw

    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss based on the prediction and target."""
        assert self.criterion.is_defined, "criterion is not defined"
        return self.criterion(prediction, target)

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        """Carry out a single training step."""
        res = self.run_step(batch, batch_idx)
        match res:
            case (y_true, y_scores):
                loss = self.compute_loss(y_scores, y_true)
            case (y_true, y_scores, loss):
                pass
            case _:
                self.error("wrong size of tuple returned by `run_step`")
                raise ValueError("wrong size of tuple returned by `run_step`")
        return self._prepare_step_output(pred=y_scores, true=y_true, loss=loss)

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        """Carry out a single validation step."""
        res = self.run_val_step(batch, batch_idx)
        match res:
            case (y_true, y_scores):
                loss = self.compute_loss(y_scores, y_true)
            case (y_true, y_scores, loss):
                pass
            case _:
                self.error("wrong size of tuple returned by `run_step`")
                raise ValueError("wrong size of tuple returned by `run_step`")
        return self._prepare_step_output(pred=y_scores, true=y_true, loss=loss)

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        """Carry out a single test step."""
        res = self.run_test_step(batch, batch_idx)
        match res:
            case (y_true, y_scores):
                loss = self.compute_loss(y_scores, y_true)
            case (y_true, y_scores, loss):
                pass
            case _:
                self.error("wrong size of tuple returned by `run_step`")
                raise ValueError("wrong size of tuple returned by `run_step`")
        return self._prepare_step_output(pred=y_scores, true=y_true, loss=loss)
