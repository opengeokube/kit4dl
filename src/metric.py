"""A module with LabelStore definition dedicated for 
collecting metrics during training or validation"""
import torchmetrics as tm


class MetricStore:
    __slots__ = ("_metrics",)

    def __init__(self, metrics: dict[str, tm.Metric]) -> None:
        self._metrics = metrics

    def reset(self):
        for met in self._metrics.values():
            met.reset()

    def update(self, true, predictions):
        for met in self._metrics.values():
            met.update(predictions, true)

    def result_dict(self) -> dict[str, float]:
        return {
            metric_name: metric.compute().item()
            for metric_name, metric in self._metrics.items()
        }
