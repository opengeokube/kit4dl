"""Module with session-related attributes."""
import sys
from typing import Any

class _ImmutableAttribute:
    _value: Any = None

    def __get__(self, obj, objtype=None) -> Any:
        return self._value

    def __set__(self, obj, value: Any) -> None:
        if self._value is not None:
            raise RuntimeError("Session properties can be set only once!")
        self._value = value


class Context:
    """Current Python session."""

    PROJECT_DIR: _ImmutableAttribute = _ImmutableAttribute()
    LOG_LEVEL: _ImmutableAttribute = _ImmutableAttribute()
    LOG_FORMAT: _ImmutableAttribute = _ImmutableAttribute()

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in Context.__dict__:
            # NOTE: maybe we can enable adding extra session arguments
            raise RuntimeError("Cannot set new session property!")
        super().__setattr__(name, value)


sys.modules[__name__] = Context() # type: ignore[assignment]
