"""A module with Kit4DL custom types."""

from typing import NewType, Literal

FullyQualifiedName = NewType("FullyQualifiedName", str)

LoggerLevel = Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
