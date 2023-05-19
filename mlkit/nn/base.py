"""A module with the base class of modules supported by MLKit"""
from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch

from mlkit.metric import MetricStore
from mlkit.nn.confmodels import Conf


class MLKitAbstractModule(ABC, pl.LightningModule):
    def __init__(self, *, conf: Conf) -> None:
        super().__init__()
        assert conf, "`conf` argument cannot be `None`"
        self._criterion: torch.nn.Module = None
        self._conf: Conf = conf
        self._setup_metrics()
        self._configure_criterion()
        self.configure(**self._conf.model.arguments)
        self.save_hyperparameters()

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure the architecture of the neural network

        Parameters
        ----------
        **kwargs : Any
            List of arguments required to setup the network architecture

        Examples
        --------
        ```python
        def setup(self, input_dims, output_dims) -> None:
            self.fc1 = nn.Sequential(
                nn.Linear(input_dims, output_dims),
            )
        ```
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
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
        def step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        raise NotImplementedError

    def val_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
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
        def step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
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
        def step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```
        """
        return self.step(batch, batch_idx)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler] | None,
    ]:
        lr_schedulers: list = self._conf.training.configured_schedulers
        return [
            self._conf.training.optimizer.optimizer(self.parameters())
        ], lr_schedulers

    def _setup_metrics(self) -> None:
        self.train_metric_tracker = MetricStore(self._conf.metrics_obj)
        self.val_metric_tracker = MetricStore(self._conf.metrics_obj)
        self.test_metric_tracker = MetricStore(self._conf.metrics_obj)

    def _configure_criterion(self) -> None:
        self._criterion = self._conf.training.criterion.criterion.to(
            self._conf.base.device
        )

    def compute_loss(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self._criterion(input, target)

    def log_train_metrics(self) -> None:
        for (
            metric_name,
            metric_value,
        ) in self.train_metric_tracker.results.items():
            self.log(
                f"train_{metric_name}",
                metric_value,
                logger=True,
            )

    def log_val_metrics(self) -> None:
        for (
            metric_name,
            metric_value,
        ) in self.val_metric_tracker.results.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                logger=True,
            )

    def log_test_metrics(self) -> None:
        for (
            metric_name,
            metric_value,
        ) in self.test_metric_tracker.results.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                logger=True,
            )

    def update_val_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor
    ) -> None:
        self.val_metric_tracker.update(true=true, predictions=predictions)

    def update_train_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor
    ) -> None:
        self.train_metric_tracker.update(true=true, predictions=predictions)

    def update_test_metrics(
        self, true: torch.Tensor, predictions: torch.Tensor
    ) -> None:
        self.test_metric_tracker.update(true=true, predictions=predictions)

    def reset_metric_trackers(self) -> None:
        self.train_metric_tracker.reset()
        self.val_metric_tracker.reset()
        self.test_metric_tracker.reset()

    def training_step(self, batch, batch_idx):
        y_true, y_scores = self.step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_scores.argmax(dim=-1)
        self.update_train_metrics(true=y_true, predictions=predictions)
        return loss

    def validation_step(self, batch, batch_idx):
        y_true, y_scores = self.val_step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_scores.argmax(dim=-1)
        self.update_val_metrics(true=y_true, predictions=predictions)
        return loss

    def test_step(self, batch, batch_idx):
        y_true, y_scores = self.test_step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_scores.argmax(dim=-1)
        self.update_test_metrics(true=y_true, predictions=predictions)
        return loss
