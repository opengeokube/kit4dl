"""Module with session-related attributes."""
import sys
from typing import Any


class _ImmutableAttribute:
    _value: Any = None
    _default: Any = None

    def __init__(self, default: Any) -> None:
        self._default = default

    def __get__(self, obj, objtype=None) -> Any:
        return self._value if self._value else self._default

    def __set__(self, obj, value: Any) -> None:
        if self._value is not None:
            raise RuntimeError("Session properties can be set only once!")
        self._value = value


# NOTE: below definitions will not be used, they are required
# just for suntax suggestions and to avoid mypy [attr-defined] error
PROJECT_DIR: Any
LOG_LEVEL: Any
LOG_FORMAT: Any
VERSION: Any


def get_dict():
    """Get dictionary of all available context-defined properties."""


class Context:
    """Current Python session."""

    PROJECT_DIR: _ImmutableAttribute = _ImmutableAttribute(default=".")
    LOG_LEVEL: _ImmutableAttribute = _ImmutableAttribute(default="INFO")
    LOG_FORMAT: _ImmutableAttribute = _ImmutableAttribute(
        default="%(asctime)s - %(levelname)s - %(message)s"
    )
    VERSION: _ImmutableAttribute = _ImmutableAttribute(default=None)

    def get_dict(self) -> dict:
        """Get dictionary of all available context-defined properties."""
        return {
            key: getattr(self, key)
            for key, value in type(self).__dict__.items()
            if isinstance(value, _ImmutableAttribute)
        }

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set the value for a context property.

        Parameters
        ----------
        name : str
            Name of the context property
        value : Any
            Value for the context property with the `key`

        Raises
        ------
        RuntimeError
            if the `name` is not defined (an attempt to create a new
            property)
        """
        if name not in Context.__dict__:
            # NOTE: maybe we can enable adding extra session arguments
            raise RuntimeError("Cannot set new session property!")
        super().__setattr__(name, value)


sys.modules[__name__] = Context()  # type: ignore[assignment]
