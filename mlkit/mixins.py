"""Module with mixing classes"""
import logging


class LoggerMixin:
    _logger: logging.Logger

    def debug(self, *args, **kwargs):
        __doc__ = self._logger.__doc__
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        __doc__ = self._logger.__doc__
        self._logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        __doc__ = self._logger.__doc__
        self._logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        __doc__ = self._logger.__doc__
        self._logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        __doc__ = self._logger.__doc__
        self._logger.critical(*args, **kwargs)
