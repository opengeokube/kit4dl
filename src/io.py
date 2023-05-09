import importlib

from src.typing import FullyQualifiedName


def get_class_from_fully_qualified_name(
    name: FullyQualifiedName | None,
) -> type | None:
    if name is None:
        return None
    if "." not in name:
        raise ValueError(
            f"expected `name` type is FullyQualifiedName. did you foget module"
            f" or package? try notation: `pkg1.module1.ClassA`"
        )
    module_name, target = name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module_name), target)
