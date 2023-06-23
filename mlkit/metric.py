"""A module with metric logic."""
import numpy as np
import torchmetrics as tm


class MetricStore:
    """Class being a store for different metrics.

    It manages resetting,updating, and computing the metrics result.
    """

    __slots__ = ("_metrics",)

    def __init__(self, metrics: dict[str, tm.Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> "MetricStore":
        """Reset metrics to the init state."""
        for met in self._metrics.values():
            met.reset()
        return self

    def update(self, true, predictions) -> "MetricStore":
        """Update metric with `true` and `predictions` values.

        Parameters
        ----------
        true : torch.Tensor
            Tensor with true labels
        predictions : torch.Tensor
            Tensor with predictions
        """
        for met in self._metrics.values():
            met.update(predictions, true)
        return self

    @property
    def results(self) -> dict[str, float]:
        """Get dictionary with metric values."""
        return {
            metric_name: (
                np.nan
                if metric._update_count  # pylint: disable=protected-access
                == 0
                else metric.compute().item()
            )
            for metric_name, metric in self._metrics.items()
        }
