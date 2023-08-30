import os
from unittest.mock import patch

import pytest

import kit4dl.io as io_


class TestIO:
    @pytest.mark.parametrize(
        "fc_name",
        [
            "torch.optim::Adam",
            "os::PathLike",
            "kit4dl.nn.base::Kit4DLAbstractModule",
        ],
    )
    def test_import_from_fully_qualified_name(self, fc_name):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(
            fc_name
        )
        assert isinstance(target_class, type)
        module, class_name = fc_name.rsplit("::")
        assert module in str(target_class)
        assert class_name in str(target_class)

    @pytest.mark.parametrize("fc_name", ["os::listdir", "numpy.linalg::eigh"])
    def test_import_from_fully_qualified_function(self, fc_name):
        func = io_.import_and_get_attr_from_fully_qualified_name(fc_name)
        assert callable(func)
        _, func = fc_name.rsplit("::")
        assert func in str(func)

    def test_import_fail_on_not_existing_file(self):
        path = r"./not_existing.py::A"
        with pytest.raises(
            AssertionError, match=r"module: ./not_existing.py does not exist"
        ):
            _ = io_.import_and_get_attr_from_fully_qualified_name(path)

    @pytest.mark.parametrize(
        "path",
        [
            "./tests/dummy_module.py::A",
            "./tests/dummy_module.py::A",
            "./tests/dummy_module2.py::T1",
        ],
    )
    def test_import_from_relative_pyfile(self, path):
        target_class = io_.import_and_get_attr_from_fully_qualified_name(path)
        assert isinstance(target_class, type)
        file, class_name = path.rsplit("::")
        assert class_name in str(target_class)
        assert os.path.exists(file)

    @pytest.mark.skipif(
        not os.path.exists(r"D:\Projekty\CMCC\kit4dl\tests\dummy_module2.py"),
        reason="file does not exists. absolute path is not correct",
    )
    def test_import_from_absolute_pyfile(self):
        path = r"D:\Projekty\CMCC\kit4dl\tests\dummy_module2.py::T1"
        target_class = io_.import_and_get_attr_from_fully_qualified_name(path)
        assert isinstance(target_class, type)
        file, class_name = path.rsplit("::")
        assert class_name in str(target_class)
        assert os.path.exists(file)

    def test_nonoverriding_imported_classes(self):
        path_1 = "./tests/dummy_module.py::T1"
        path_2 = "./tests/dummy_module2.py::T1"

        cls1 = io_.import_and_get_attr_from_fully_qualified_name(path_1)
        cls2 = io_.import_and_get_attr_from_fully_qualified_name(path_2)

        assert id(cls1) != id(cls2)
        obj1 = cls1()
        assert hasattr(obj1, "f1")
        assert not hasattr(obj1, "f2")

        obj2 = cls2()
        assert hasattr(obj2, "f2")
        assert not hasattr(obj2, "f1")
