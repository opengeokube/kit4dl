"""Configuration formatting utils."""
import os

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    UndefinedError,
)

_WINDOWS_PATHNAME_SEP = "\\"
_UNIX_PATHNAME_SEP = "/"


def substitute_symbols(path: str, **kwargs) -> str:
    """Replace placeholders with the provided keyword arguments and env vars.

    The function replaces `load` text with the provided keyword arguments
    and, eventually, all available environmental variables.

    Parameters
    ----------
    path : str
        Path to the file with placeholders
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
    >>> load = "hi {{ name }}"
    >>> substitute_symbols(load, name="John")
    "hi John"
    ```

    ```python
    >>> load = "my env var has value {{ env['MY_ENV_VAR'] }}"
    >>> substitute_symbols(load)
    "my env var has value /usr/dataset"
    ```

    ```python
    >>> load = "hi {{ name }}"
    >>> substitute_symbols(load, age=10)
    KeyError no value found for the placeholder `name`
    ```
    """
    assert path is not None, "`path` cannot be `None`!"
    base_path, filename = os.path.split(path)
    tmpl_env = Environment(
        loader=FileSystemLoader(base_path), undefined=StrictUndefined
    )
    try:
        return tmpl_env.get_template(filename).render(env=os.environ, **kwargs)
    except UndefinedError as err:
        raise ValueError(err.message) from err


def escape_os_sep(load: str) -> str:
    """Escape OS-specific directory separator."""
    # NOTE: we make os separators uniform across the text load
    escaped = load.replace(_WINDOWS_PATHNAME_SEP, os.sep)
    escaped = escaped.replace(_UNIX_PATHNAME_SEP, os.sep)
    return escaped.replace(
        _WINDOWS_PATHNAME_SEP,
        "".join([_WINDOWS_PATHNAME_SEP, _WINDOWS_PATHNAME_SEP]),
    )
