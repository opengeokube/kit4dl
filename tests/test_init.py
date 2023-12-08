import pytest
import numpy as np

from kit4dl import _overwrite_dict


class TestInit:
    @pytest.fixture
    def conf_dict(self) -> dict:
        return {"a": {"b": {"c": 100}}, "d": 50}

    def test_overwrite_root_level_key(self, conf_dict):
        _overwrite_dict(conf_dict, {"a.d": 100})
        assert conf_dict["a"]["d"] == 100

    def test_overwrite_multiple_level_key(self, conf_dict):
        _overwrite_dict(conf_dict, {"a.b.c": 500})
        assert conf_dict["a"]["b"]["c"] == 500
