import os
import shutil

import pytest

from mlkit.cli.app import init, train


@pytest.fixture
def project_name():
    return "new-project-1"


@pytest.fixture(autouse=True, scope="function")
def clear_new_project(project_name):
    shutil.rmtree(project_name, ignore_errors=True)
    yield
    shutil.rmtree(project_name, ignore_errors=True)


class TestApp:
    def test_init_creates_dir_with_name(self, project_name):
        init(name=project_name)
        assert os.path.exists(project_name)

    def test_init_creates_dir_default_name(self):
        init()
        assert os.path.exists("new_mlkit_project")
        shutil.rmtree("new_mlkit_project")

    def test_init_just_three_files_created(self, project_name):
        init(name=project_name)
        files = os.listdir(project_name)
        assert len(files) == 3
        assert "dataset.py" in files
        assert "model.py" in files
        assert "conf.toml" in files
