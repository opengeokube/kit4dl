"""A module with the base class of modules supported by Kit4DL."""
import logging
from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl
import torch

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from kit4dl.metric import MetricStore
from kit4dl.mixins import LoggerMixin
from kit4dl.nn.confmodels import Conf
from kit4dl.stages import Stage


class StepOutput(dict):
    """Output of the single train/val/test step."""

    def __init__(
        self, *, pred: torch.Tensor, true: torch.Tensor, loss: torch.Tensor
    ):
        super().__init__()
        self["pred"] = pred
        self["true"] = true
        self["loss"] = loss

    @property
    def loss(self) -> torch.Tensor:
        """Get loss value."""
        return self["loss"]

    @property
    def predictions(self) -> torch.Tensor:
        """Get predictions."""
        return self["pred"]

    @property
    def labels(self) -> torch.Tensor:
        """Get ground-turth labels."""
        return self["true"]


class Kit4DLAbstractModule(
    ABC, pl.LightningModule, LoggerMixin
):  # pylint: disable=too-many-ancestors
    """Base abstract class for Kit4DL modules."""

    def __init__(self, *, conf: Conf) -> None:
        super().__init__()
        assert conf, "`conf` argument cannot be `None`"
        self._criterion: torch.nn.Module | None = None
        self._conf: Conf = conf

        self._logger = logging.getLogger("lightning")
        self.configure_logger()

        self._configure_metrics()
        self._configure_criterion()
        self.configure(**self._conf.model.arguments)
        self.save_hyperparameters(self._conf.dict())

    def configure_logger(self) -> None:
        """Configure logger based on the configuration passed to the class.

        The methods configure the logger format and sets it to all
        the handlers.
        """
        self._logger.setLevel(self._conf.logging.level)  # type: ignore[arg-type]
        if self._conf.logging.format_:
            formatter = logging.Formatter(self._conf.logging.format_)
            for handler in self._logger.handlers:
                handler.setFormatter(formatter)
        for handler in self._logger.handlers:
            handler.setLevel(self._conf.logging.level)  # type: ignore[arg-type]

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

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LRScheduler] | None,]:
        """Configure optimizers and schedulers."""
        self.debug("configuring optimizers and lr epoch schedulers...")
        optimizer: torch.optim.Optimizer = (
            self._conf.training.optimizer.optimizer(self.parameters())
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

    def _configure_metrics(self) -> None:
        self.debug("configuring metrics...")
        self.train_metric_tracker = MetricStore(self._conf.metrics_obj)
        self.val_metric_tracker = MetricStore(self._conf.metrics_obj)
        self.test_metric_tracker = MetricStore(self._conf.metrics_obj)

    def _configure_criterion(self) -> None:
        self.debug("configuring criterion...")
        self._criterion = self._conf.training.criterion.criterion.to(
            self._conf.base.device
        )
        self.info("selected criterion is: %s", self._criterion)

    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss based on the prediction and target."""
        assert self._criterion, "criterion is None"
        return self._criterion(prediction, target)

    def log_train_metrics(self) -> None:
        """Log train metrics."""
        for (
            metric_name,
            metric_value,
        ) in self.train_metric_tracker.results.items():
            stage_metric_name = f"{Stage.TRAIN}_{metric_name}"
            self.info(
                "epoch: %d metric: %s value: %s",
                self.current_epoch,
                stage_metric_name,
                metric_value,
            )
            self.log(
                stage_metric_name,
                metric_value,
                logger=True,
            )

    def log_val_metrics(self) -> None:
        """Log validation metrics."""
        for (
            metric_name,
            metric_value,
        ) in self.val_metric_tracker.results.items():
            stage_metric_name = f"{Stage.VALIDATION}_{metric_name}"
            self.info(
                "epoch: %d metric: %s value: %s",
                self.current_epoch,
                stage_metric_name,
                metric_value,
            )
            self.log(
                stage_metric_name,
                metric_value,
                logger=True,
            )

    def log_test_metrics(self) -> None:
        """Log test metrics."""
        for (
            metric_name,
            metric_value,
        ) in self.test_metric_tracker.results.items():
            stage_metric_name = f"{Stage.TEST}_{metric_name}"
            self.info(
                "epoch: %d metric: %s value: %s",
                self.current_epoch,
                stage_metric_name,
                metric_value,
            )
            self.log(
                stage_metric_name,
                metric_value,
                logger=True,
            )

    def update_train_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor, loss: torch.Tensor
    ) -> None:
        """Update train metrics with true and prediction values."""
        self.train_metric_tracker.update(true=true, predictions=predictions)
        if loss:
            self.log(name=f"{Stage.TRAIN}_loss", value=loss, logger=True)

    def update_val_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor, loss: torch.Tensor
    ) -> None:
        """Update validation metrics with true and prediction values."""
        self.val_metric_tracker.update(true=true, predictions=predictions)
        if loss:
            self.log(name=f"{Stage.VALIDATION}_loss", value=loss, logger=True)

    def update_test_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor, loss: torch.Tensor
    ) -> None:
        """Update test metrics with true and prediction values."""
        self.test_metric_tracker.update(true=true, predictions=predictions)
        if loss:
            self.log(name=f"{Stage.TEST}_loss", value=loss, logger=True)

    def reset_metric_trackers(self) -> None:
        """Reset all metric trackers: train, validation, and test."""
        self.train_metric_tracker.reset()
        self.val_metric_tracker.reset()
        self.test_metric_tracker.reset()

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
        self.update_train_metrics(true=y_true, predictions=y_scores, loss=loss)
        return StepOutput(pred=y_scores, true=y_true, loss=loss)

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
        self.update_val_metrics(true=y_true, predictions=y_scores, loss=loss)
        return StepOutput(pred=y_scores, true=y_true, loss=loss)

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
        self.update_test_metrics(true=y_true, predictions=y_scores, loss=loss)
        return StepOutput(pred=y_scores, true=y_true, loss=loss)
