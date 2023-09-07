import os
import shutil

import pytest

from kit4dl.cli.app import init, train


@pytest.fixture
def experiment_name():
    return "new-project-1"


@pytest.fixture(autouse=True, scope="function")
def clear_new_project(experiment_name):
    shutil.rmtree(experiment_name, ignore_errors=True)
    yield
    shutil.rmtree(experiment_name, ignore_errors=True)


class TestApp:
    def test_init_creates_dir_with_name(self, experiment_name):
        init(name=experiment_name)
        assert os.path.exists(experiment_name)

    def test_init_creates_dir_default_name(self):
        init()
        assert os.path.exists("new_kit4dl_project")
        shutil.rmtree("new_kit4dl_project")

    def test_init_just_three_files_created(self, experiment_name):
        init(name=experiment_name)
        files = os.listdir(experiment_name)
        assert len(files) == 3
        assert "datamodule.py" in files
        assert "model.py" in files
        assert "conf.toml" in files
