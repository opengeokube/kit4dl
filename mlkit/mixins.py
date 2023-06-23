"""Module with mixing classes definitions."""
import logging


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

    def debug(self, *args, **kwargs):
        """Log on debug level."""
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        """Log on info level."""
        self._logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        """Log on warning level."""
        self._logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        """Log on error level."""
        self._logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        """Log on critical level."""
        self._logger.critical(*args, **kwargs)
