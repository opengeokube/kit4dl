"""Module with mixing classes definitions."""

import logging

from kit4dl.kit4dl_types import LoggerLevel


class LoggerMixin:
    """
    Mixing class for direct logging.

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
