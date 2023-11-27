"""Module with mixing classes definitions."""

import os
from typing import Any, ClassVar
import logging

from pydantic import BaseModel

from kit4dl.kit4dl_types import LoggerLevel


class ObfuscateKeyMixing:
    """
    Mixin class for obfuscating sensitive data.

    The class provides `obfuscated_dict` method that return dict-like
    `pydantic.BaseModel`- representation whose values are obfuscated
    for keys containing the predefined value taken from
    `KIT4DL_KEY_TO_OBFUSCATE` environmental variable or `key` by default.
    Obfuscating value is taken from KIT4DL_OBFUSCATING_VALUE environmental
    variable (`***` by default).
    """

    KIT4DL_KEY_TO_OBFUSCATE: ClassVar[str]
    KIT4DL_OBFUSCATING_VALUE: ClassVar[str]

    def __new__(cls, *arr, **kw):  # pylint: disable=unused-argument
        """Create a ObfuscateKeyMixing instance. Insitialize class vars."""
        assert issubclass(cls, BaseModel), (
            "Class with `ObfuscateKeyMixing` needs to be subclass of"
            " `pydantic.BaseModel`"
        )
        ObfuscateKeyMixing.KIT4DL_KEY_TO_OBFUSCATE = os.environ.get(
            "KIT4DL_KEY_TO_OBFUSCATE", "key"
        )
        ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE = os.environ.get(
            "KIT4DL_OBFUSCATING_VALUE", "***"
        )
        return super().__new__(cls)

    @staticmethod
    def _recursively_obfuscate_key_inplace(
        mapping: dict, key_to_obfuscate: str
    ):
        for key in mapping.keys():
            if isinstance(mapping[key], dict):
                mapping[key] = (
                    ObfuscateKeyMixing._recursively_obfuscate_key_inplace(
                        mapping[key], key_to_obfuscate
                    )
                )
            if key_to_obfuscate in key:
                mapping[key] = ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE
        return mapping

    def obfuscated_dict(self, *arr, **kw) -> dict[str, Any]:
        """Obfuscate values for fix keys in the model dict representation."""
        model_dict = BaseModel.model_dump(self, *arr, **kw)  # type: ignore[arg-type]
        return ObfuscateKeyMixing._recursively_obfuscate_key_inplace(
            model_dict.copy(),
            key_to_obfuscate=ObfuscateKeyMixing.KIT4DL_KEY_TO_OBFUSCATE,
        )


class LoggerMixin:
    """
    Mixin class for direct logging.

    The class provides methods for direct logging instead of
    calling `_logger` methods.`
    The methods have the same signatres as those defined in the
    `logging.Logger` class.

    Note
    ----
    The class using `LoggerMixin` need to have `_logger` defined.

    Examples
    --------
    ```
    >>> class MyClass(LoggerMixin): pass
    >>> MyClass().debug("some debug message")
    """

    _logger: logging.Logger

    def configure_logger(
        self,
        name: str,
        level: LoggerLevel | None,
        logformat: str | None = None,
    ) -> None:
        """Configure logger."""
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)  # type: ignore[arg-type]
        for handler in self._logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                break
        else:
            self._logger.addHandler(logging.StreamHandler())
        if logformat:
            formatter = logging.Formatter(logformat)
            for handler in self._logger.handlers:
                handler.setFormatter(formatter)
        for handler in self._logger.handlers:
            handler.setLevel(level)  # type: ignore[arg-type]

    def debug(self, *args, **kwargs):
        """Log on debug level."""
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        """Log on info level."""
        self._logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        """Log on warning level."""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """Log on error level."""
        self._logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        """Log on critical level."""
        self._logger.critical(*args, **kwargs)
