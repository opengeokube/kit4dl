import pytest

import toml

from src.nn.confmodels import MetricsConf


@pytest.fixture
def metric_dict() -> dict:
    load = """
[metrics]
Precision = {task = "multiclass", num_classes=10}
FBetaScore = {task = "multiclass", num_classes=10, beta = 2}
    """
    return toml.loads(load)


def test_parse_metrics_present(metric_dict):
    metrics = MetricsConf(**metric_dict)
    breakpoint()
    assert "Precision" in metrics
    assert "FBetaScore" in metrics
    assert metrics["Precision"] == {"task": "multiclass", "num_classes": 10}
    assert metrics["FBetaScore"] == {
        "task": "multiclass",
        "num_classes": 10,
        "beta": 2,
    }


def test_parse_metrics_fail_on_non_existing():
    load = """
[metrics]
Abc = {task = "multiclass", num_classes=10}
    """
    metric_dict = toml.loads(load)
    with pytest.raises(KeyError, match=r"metric `Abc`*"):
        _ = MetricsConf(**metric_dict)
