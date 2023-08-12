import warnings
from contextlib import suppress
from functools import partial

import numpy as np
import pytest
import sklearn.metrics as smt
import torch
import torchmetrics as tm

from kit4dl.metric import MetricStore


class TestMetricStore:
    @pytest.fixture
    def metric_dict(self):
        yield {
            "Precision": tm.Precision(task="binary"),
            "Recall": tm.Recall(task="binary"),
        }

    @pytest.fixture
    def metric_store(self, metric_dict):
        yield MetricStore(metric_dict)

    def test_metricstore_results_repeatable(self, metric_store):
        metric_store.update(torch.LongTensor([0, 1]), torch.LongTensor([0, 1]))
        metric_store.update(torch.LongTensor([0, 1]), torch.LongTensor([1, 0]))
        res1 = metric_store.results
        res2 = metric_store.results
        assert id(res1) != id(res2)
        assert res1 == res2

    def test_get_empty_result_nan(self, metric_store):
        assert metric_store.results == {"Precision": np.nan, "Recall": np.nan}

    def test_reset_if_zeros_out(self, metric_store):
        metric_store.results == {"Precision": np.nan, "Recall": np.nan}
        metric_store.update(
            torch.randint(0, 2, size=(100,)), torch.randint(0, 2, size=(100,))
        )
        assert metric_store.results["Precision"] != 0.0
        assert metric_store.results["Recall"] != 0.0
        metric_store.reset()
        metric_store.results == {"Precision": np.nan, "Recall": np.nan}

    def test_metric_aggregation(self, metric_store):
        true = torch.randint(0, 2, size=(100,))
        true2 = torch.randint(0, 2, size=(100,))
        pred = torch.randint(0, 2, size=(100,))
        pred2 = torch.randint(0, 2, size=(100,))
        metric_store.update(true, pred)
        metric_store.update(true2, pred2)
        metrics_1 = metric_store.results

        metric_store.update(torch.cat([true, true2]), torch.cat([pred, pred2]))
        metrics_2 = metric_store.results
        assert metrics_1 == metrics_2

    def test_no_warning_on_before_update(self, metric_store):
        with warnings.catch_warnings(record=True) as warns:
            _ = metric_store.results
            assert len(warns) == 0
