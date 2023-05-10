"""A module with input-output operations"""
import importlib
import os
import sys
from typing import Any

from mlkit.typing import FullyQualifiedName


def import_and_get_attr_from_fully_qualified_name(
    name: FullyQualifiedName | str | os.PathLike,
) -> Any:
    assert name is not None, "`name` argument cannot be `None`"
    if "::" not in name:
        raise ValueError(
            f"missing `::` in the name: `{name}`. remember to put the double"
            " colon `::` between: \n a) the fully qualified name and the"
            " class, like `package.module::MyClass` \n 2) the module path and"
            " the class, like `/usr/my_user/my_module.py::MyClass`"
        )
    path, attr_name = name.rsplit("::", maxsplit=1)
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location(
            "_imported_nn_module", path
        )
        _imported_nn_module = importlib.util.module_from_spec(spec)
        sys.modules["_imported_nn_module"] = _imported_nn_module
        spec.loader.exec_module(_imported_nn_module)
        return getattr(_imported_nn_module, attr_name)
    elif importlib.util.find_spec(path):
        return getattr(importlib.import_module(path), attr_name)
    else:
        raise ModuleNotFoundError(
            f"module defined as `{path}` cannot be imported or found in the"
            " system"
        )
