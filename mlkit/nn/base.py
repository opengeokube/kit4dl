"""A module with the base class of modules supported by MLKit."""
import logging
from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl
import torch

from mlkit.metric import MetricStore
from mlkit.mixins import LoggerMixin
from mlkit.nn.confmodels import Conf


class MLKitAbstractModule(
    ABC, pl.LightningModule, LoggerMixin
):  # pylint: disable=too-many-ancestors
    """Base abstract class for MLKit modules."""

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
        self.save_hyperparameters()

    def configure_logger(self) -> None:
        """Configure logger based on the configuration passed to the class.

        The methods configure the logger format and sets it to all
        the handlers.
        """
        if self._conf.base.log_format:
            formatter = logging.Formatter(self._conf.base.log_format)
            for handler in self._logger.handlers:
                handler.setFormatter(formatter)
        for handler in self._logger.handlers:
            handler.setLevel(self._conf.base.log_level)  # type: ignore[arg-type]

    @property
    def _mlkit_logger(self) -> logging.Logger:
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
    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
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
            A tuple whose 1st element is `torch.Tensor` of ground-truth labels
            and the 2nd - output of the network

        Examples
        --------
        ```python
        ...
        def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        raise NotImplementedError

    def run_val_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry out single validation step for the given `batch`.

        Return a tuple of two `torch.Tensor`'s: true labels and predicted scores.
        If not overriden, the implementation of `step` method is used.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch

        Returns
        -------
        result : tuple of `torch.Tensor`
            A tuple whose 1st element is `torch.Tensor` of ground-truth labels
            and the 2nd - output of the network

        Examples
        --------
        ```python
        ...
        def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        return self.run_step(batch, batch_idx)

    def run_test_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry out single test step for the given `batch`.

        Return a tuple of two `torch.Tensor`'s: true labels and predicted scores.
        If not overriden, the implementation of `step` method is used.

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch

        Returns
        -------
        result : tuple of `torch.Tensor`
            A tuple whose 1st element is `torch.Tensor` of ground-truth labels
            and the 2nd - output of the network

        Examples
        --------
        ```python
        ...
        def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        return self.run_step(batch, batch_idx)

    def run_predict_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        Examples
        --------
        ```python
        ...
        def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        return self.run_step(batch, batch_idx)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler] | None,
    ]:
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
        self.info("selected critarion is: %s", self._criterion)

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
            stage_metric_name = f"train_{metric_name}"
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
            stage_metric_name = f"val_{metric_name}"
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
            stage_metric_name = f"test_{metric_name}"
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

    def update_val_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor
    ) -> None:
        """Update validation metrics with true and prediction values."""
        self.val_metric_tracker.update(true=true, predictions=predictions)

    def update_train_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor
    ) -> None:
        """Update train metrics with true and prediction values."""
        self.train_metric_tracker.update(true=true, predictions=predictions)

    def update_test_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor
    ) -> None:
        """Update test metrics with true and prediction values."""
        self.test_metric_tracker.update(true=true, predictions=predictions)

    def reset_metric_trackers(self) -> None:
        """Reset all metric trackers: train, validation, and test."""
        self.train_metric_tracker.reset()
        self.val_metric_tracker.reset()
        self.test_metric_tracker.reset()

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        """Carry out a single training step."""
        y_true, y_scores = self.run_step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_scores.argmax(dim=-1)
        self.update_train_metrics(true=y_true, predictions=predictions)
        return loss

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ
        """Carry out a single validation step."""
        y_true, y_scores = self.run_val_step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_scores.argmax(dim=-1)
        self.update_val_metrics(true=y_true, predictions=predictions)
        return loss

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        """Carry out a single test step."""
        y_true, y_scores = self.run_test_step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_scores.argmax(dim=-1)
        self.update_test_metrics(true=y_true, predictions=predictions)
        return loss
