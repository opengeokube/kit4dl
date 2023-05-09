"""A module with the base class of modules supported by MLKit"""
from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch

from src.metric import MetricStore
from src.nn.confmodels import Conf


class MLKitAbstractModule(ABC, pl.LightningModule):
    conf: Conf
    criterion: torch.nn.Module = None

    def __init__(self, *, conf: Conf) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self._setup_metrics()
        self._configure_criterion()

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler] | None,
    ]:
        assert self.conf is not None, (
            "configuration was not set. did you forget to call"
            " `obj.setup_configuration(conf)` method?"
        )
        # TODO: handle list of epoch schedulers
        lr_schedulers = None
        return [
            self.conf.training.optimizer.optimizer(self.parameters())
        ], lr_schedulers

    def _setup_metrics(self) -> None:
        if not self.conf:
            return None
        self.train_metric_tracker = MetricStore(self.conf.metrics_obj)
        self.val_metric_tracker = MetricStore(self.conf.metrics_obj)
        self.test_metric_tracker = MetricStore(self.conf.metrics_obj)

    def _configure_criterion(self) -> None:
        if not self.conf:
            return None
        self.criterion = self.conf.training.criterion.criterion.to(
            self.conf.base.device
        )

    def compute_loss(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        assert self.criterion, "critarion was not configured!"
        return self.criterion(input, target)

    def log_train_metrics(self) -> None:
        for (
            metric_name,
            metric_value,
        ) in self.train_metric_tracker.result_dict().items():
            self.log(
                f"train_{metric_name}",
                metric_value,
                logger=True,
            )

    def log_val_metrics(self) -> None:
        for (
            metric_name,
            metric_value,
        ) in self.val_metric_tracker.result_dict().items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                logger=True,
            )

    def log_test_metrics(self) -> None:
        for (
            metric_name,
            metric_value,
        ) in self.test_metric_tracker.result_dict().items():
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

    @abstractmethod
    def step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Carry out single train or validation step for the given `batch`.
        Return a tuple of two `torch.Tensor`'s: true labels and predicted scores.
        Sample code could look like the below:

        ```python
        ...
        def step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            feature_input, label_input = batch
            scores = self(feature_input)
            return (label_input, scores)
        ```

        Parameters
        ----------
        batch : torch.Tensor or tuple of torch.Tensor or list of torch.Tensor
            The output of the Dataloader
        batch_idx : int
            Index of the batch
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        y_true, y_scores = self.step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_true.argmax(dim=-1)
        self.update_train_metrics(true=y_true, predictions=predictions)
        return loss

    def validation_step(self, batch, batch_idx):
        y_true, y_scores = self.step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_true.argmax(dim=-1)
        self.update_val_metrics(true=y_true, predictions=predictions)
        return loss

    def test_step(self, batch, batch_idx):
        y_true, y_scores = self.step(batch, batch_idx)
        loss = self.compute_loss(y_scores, y_true)
        predictions = y_true.argmax(dim=-1)
        self.update_test_metrics(true=y_true, predictions=predictions)
        return loss
