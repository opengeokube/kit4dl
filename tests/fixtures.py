import os

import pytest
import toml

from mlkit.nn.confmodels import Conf


@pytest.fixture
def true_conf():
    yield Conf(
        root_dir=os.path.join(os.getcwd(), "tests", "resources"),
        **toml.load(os.path.join("tests", "resources", "test_file.toml")),
    )
