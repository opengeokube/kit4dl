"""Configuration formatting utils."""
import os
import string

_WINDOWS_PATHNAME_SEP = "\\"
_UNIX_PATHNAME_SEP = "/"


def substitute_symbols(load: str, **kwargs) -> str:
    """Replace placeholders with the provided keyword arguments.

    Parameters
    ----------
    load : str
        String with placeholders
    **kwargs : Any
        Keword arguments to replace in `load`

    Returns
    -------
    replaced : str
        String with replaced placeholders

    Raises
    ------
    KeyError
        if a value was specified for any placeholder in `load`

    Examples
    --------
    ```python
    >>> load = "hi {name}"
    >>> substitute_symbols(load, name="John")
    "hi John"
    ```

    ```python
    >>> load = "hi {name}"
    >>> substitute_symbols(load, age=10)
    KeyError no value found for the placeholder `name`
    ```
    """
    assert load is not None, "`load` cannot be `None`!"
    tmpl = string.Template(load)
    return tmpl.substitute(**kwargs)


def escape_os_sep(load: str) -> str:
    """Escape OS-specific directory separator."""
    # NOTE: we make os separators uniform across the text load
    escaped = load.replace(_WINDOWS_PATHNAME_SEP, os.sep)
    escaped = escaped.replace(_UNIX_PATHNAME_SEP, os.sep)
    return escaped.replace(
        _WINDOWS_PATHNAME_SEP,
        "".join([_WINDOWS_PATHNAME_SEP, _WINDOWS_PATHNAME_SEP]),
    )
