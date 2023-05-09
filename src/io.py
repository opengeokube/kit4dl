"""A module with input-output operations"""
import importlib

from src.typing import FullyQualifiedName


def get_class_from_fully_qualified_name(
    name: FullyQualifiedName,
) -> type:
    assert name is not None, "`name` argument cannot be `None`"
    if "." not in name:
        raise ValueError(
            "expected `name` type is FullyQualifiedName. did you foget module"
            " or package? try notation: `pkg1.module1.ClassA`"
        )
    module_name, target = name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module_name), target)
