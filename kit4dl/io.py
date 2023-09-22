"""A module with input-output operations."""
import importlib
import os
import sys
from typing import Any

from kit4dl.kit4dl_types import FullyQualifiedName

_TARGET_SPLIT = "::"
_MODULE_SEP = ".py::"
PROJECT_DIR = "PROJECT_DIR"


def _does_path_contain_module_file(target: str | FullyQualifiedName):
    return target.strip().endswith(".py") or _MODULE_SEP in target


def split_target(target: str | FullyQualifiedName) -> tuple[str, str]:
    """Split target name or fully qualified name by `::` to get the path and the attribute.

    Parameters
    ----------
    target : str or FullyQualifiedName
        The value of the class or Python module path

    Returns
    -------
    path_attr : tuple of str
        A tuple whose first item is a path component of the target and the second - attribute name

    Example
    -------
    ```
    >>> split_target("a.b.c::D")
    "a.b.c", "D"
    ```
    """
    path, attr_name = target.rsplit(_TARGET_SPLIT, maxsplit=1)
    return (path, attr_name)


def connect_target(path: str, attr_name: str) -> str:
    """Connect path and attribute name with `::`.

    Parameters
    ----------
    path : str
        The path
    attr_name: str
        The attribute name

    Returns
    -------
    target : str
        A string of `path` and `attr_name` connected with `::`

    Example
    -------
    ```
    >>> connect_target("a.b.c", "D")
    "a.b.c::D"
    ```
    """
    return _TARGET_SPLIT.join([path, attr_name])


def maybe_get_abs_target(
    target: str | FullyQualifiedName, root_dir: str
) -> str:
    """Get absolute path for the target.

    If `target` provided as the absolut path or as the fully qualified name
    of a module, no changes is done and `target` arguments is returned
    directly. If `target` is a relative path to a Python module, it is
    converted to the absolute path-based target.

    Parameters
    ----------
    target : str or FullyQualifiedName
        The value of the class or Python module path
    root_dir : str
        Root dir with respect to which absolut path should be computed

    Returns
    -------
    abs_path : str
        Absolute path for the provided target

    Example
    -------
    ```
    >>> maybe_get_abs_target("a.py::MyClass", "/work/usr")
    "/work/usr/a.py::MyClass"
    ```

    ```
    >>> maybe_get_abs_target("os::PathLike", "/work/usr")
    os::PathLike
    ```

    """
    if not _does_path_contain_module_file(target):
        return target
    path, attr_name = split_target(target)
    if os.path.isabs(path):
        return target
    return connect_target(os.path.join(root_dir, path), attr_name)


def import_module_from_file(path: str, exec_module: bool = False):
    """Import module from file indicated by the relative path."""
    assert _does_path_contain_module_file(
        path
    ), f"path: {path} is not a Python module"
    assert os.path.exists(path), f"module: {path} does not exist"
    spec = importlib.util.spec_from_file_location(
        "_file_imported_module", path
    )
    if spec is None:
        raise RuntimeError(f"module {path} is not defined")
    _file_imported_module = importlib.util.module_from_spec(spec)
    if exec_module:
        spec.loader.exec_module(_file_imported_module)  # type: ignore[union-attr]
    return _file_imported_module


def get_class_from_py_file(path: str, name: str):
    """Get class defined in the Python file.

    Parameters
    ----------
    path : str
        The path to the Python file where the class is defined
    name : str
        The name of the class defined in the Python module indicated by the `path` argument

    Returns
    -------
    clss : type
        A class loaded from the Python file

    Example
    -------
    ```
    >>> et_class_from_py_file("./my_module.py", "MyClass")
    <class '_file_imported_module.MyClass'>
    ```
    """
    _file_imported_module = import_module_from_file(
        path=path, exec_module=True
    )
    sys.modules["_file_imported_module"] = _file_imported_module
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
        if `name` format is wrong (does not contain class separator `::`)
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
    if _does_path_contain_module_file(path):
        return get_class_from_py_file(path, attr_name)
    if importlib.util.find_spec(path):
        return getattr(importlib.import_module(path), attr_name)
    raise ModuleNotFoundError(
        f"module defined as `{path}` cannot be imported or found in the system"
    )
