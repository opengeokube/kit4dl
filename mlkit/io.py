"""A module with input-output operations"""
import importlib
import os
import sys
from typing import Any

from mlkit.types import FullyQualifiedName

_TARGET_SPLIT = "::"


def split_target(target: str | FullyQualifiedName) -> tuple[str, str]:
    path, attr_name = target.rsplit(_TARGET_SPLIT, maxsplit=1)
    return (path, attr_name)


def connect_target(path: str, attr_name: str) -> str:
    return _TARGET_SPLIT.join([path, attr_name])


def maybe_get_abs_target(target: str, root_dir: str | os.PathLike) -> str:
    if ".py" not in target:
        return target
    path, attr_name = split_target(target)
    if os.path.isabs(path):
        return target
    return connect_target(os.path.join(root_dir, path), attr_name)


def get_class_from_py_file(path: str | os.PathLike, name: str):
    assert ".py" in path, f"path: {path} is not a Python module"
    assert os.path.exists(path), f"module: {path} does not exist"
    spec = importlib.util.spec_from_file_location(
        "_file_imported_module", path
    )
    if not spec:
        raise RuntimeError(f"module {path} is not defined")
    _file_imported_module = importlib.util.module_from_spec(spec)
    sys.modules["_file_imported_module"] = _file_imported_module
    spec.loader.exec_module(_file_imported_module)
    return getattr(_file_imported_module, name)


def import_and_get_attr_from_fully_qualified_name(
    name: FullyQualifiedName | str,
) -> Any:
    """Take class based on fully qualified name or module path.

    Parameters
    ----------
    name :  FullyQualifiedName or str
        Fully qualified name to the class, i.e. my_package.my_module::MyClass
        or path (absolute or relative) to the module together with class name:
        my_module.py::MyClass

    Returns
    -------
    res_class : type
        A type from the available module or imported from a .py file

    Raises
    ------
    ValueError
        if `name` format is wrong
    ModuleNotFoundError
        if a module (installed or from file) was not found

    Examples
    --------
    ```bash
    >>> import_and_get_attr_from_fully_qualified_name("sklearn.ensemble::RandomForestClassifier")
    <class 'sklearn.ensemble._forest.RandomForestClassifier'>
    ```

    ```bash
    >>> import_and_get_attr_from_fully_qualified_name("./dataset::MyMNISTDatamodule")
    <class '_file_imported_module.MyMNISTDatamodule'>
    ```
    """
    assert name is not None, "`name` argument cannot be `None`"
    if "::" not in name:
        raise ValueError(
            f"missing `::` in the name: `{name}`. remember to put the double"
            " colon `::` between: \n a) the fully qualified name and the"
            " class, like `package.module::MyClass` \n 2) the module path and"
            " the class, like `/usr/my_user/my_module.py::MyClass`"
        )
    path, attr_name = split_target(name)
    if ".py" in path:
        return get_class_from_py_file(path, attr_name)
    elif importlib.util.find_spec(path):
        return getattr(importlib.import_module(path), attr_name)
    else:
        raise ModuleNotFoundError(
            f"module defined as `{path}` cannot be imported or found in the"
            " system"
        )
