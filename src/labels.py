import torchmetrics as tm


class LabelsStore:
    def __init__(self, metrics: dict[str, dict]) -> None:
        self._metrics = {}
        for metric_name, kwargs in metrics.items():
            self._metrics[metric_name] = getattr(tm, metric_name)(**kwargs)

    def reset(self):
        for m in self._metrics.values():
            m.reset()

    def update(self, true, logits):
        for m in self._metrics.values():
            m.update(logits, true)

    def result_dict(self) -> dict:
        return {
            metric_name: metric.compute().item()
            for metric_name, metric in self._metrics.items()
        }
