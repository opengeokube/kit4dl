"""A module with LabelStore definition dedicated for 
collecting metrics during training or validation"""
import torchmetrics as tm


class LabelsStore:
    def __init__(self, metrics: dict[str, dict]) -> None:
        self._metrics = {}
        for metric_name, kwargs in metrics.items():
            self._metrics[metric_name] = getattr(tm, metric_name)(**kwargs)

    def reset(self):
        for met in self._metrics.values():
            met.reset()

    def update(self, true, logits):
        for met in self._metrics.values():
            met.update(logits, true)

    def result_dict(self) -> dict:
        return {
            metric_name: metric.compute().item()
            for metric_name, metric in self._metrics.items()
        }
