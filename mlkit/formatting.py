"""Configuration formatting utils."""
import string

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
    fmt = string.Formatter()
    for _, placeholder, _, _ in fmt.parse(load):
        if placeholder and placeholder not in kwargs:
            raise KeyError(f"no value found for the placeholder `{placeholder}`")
    return fmt.format(load, **kwargs)