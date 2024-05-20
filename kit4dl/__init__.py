"""Kit4DL package."""

__all__ = (
    "setup_and_train",
    "setup_and_test",
)
import os
import sys
import logging

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml  # type: ignore[no-redef]


from lightning.pytorch.callbacks import Callback as Kit4DLCallback
from kit4dl import context
from kit4dl._version import __version__
import kit4dl.nn.confmodels
from kit4dl.formatting import escape_os_sep, substitute_symbols
from kit4dl.nn.trainer import Trainer
from kit4dl.nn.base import Kit4DLAbstractModule
from kit4dl.nn.dataset import Kit4DLAbstractDataModule


# ##############################
#       CONFIGURE LOGGER
# ##############################
_LOG_LEVEL = "INFO"
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logger(
    logger: logging.Logger,
    level: str = "INFO",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # pylint: disable=redefined-builtin
):
    """Configure logger with the given level and format."""
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    for hdl in log.handlers:
        hdl.setFormatter(formatter)
        hdl.setLevel(level)  # type: ignore[arg-type]


log = logging.getLogger("Kit4DL")
configure_logger(log, _LOG_LEVEL, _LOG_FORMAT)


def get_default_conf_path() -> str:
    """Get default conf file (in the current working dir)."""
    return os.path.join(os.getcwd(), "conf.toml")


def setup_and_train(
    conf_path: str = get_default_conf_path(), overwrite: dict | None = None
) -> kit4dl.nn.confmodels.Conf:
    """Set up experiment and run train/val.

    Parameters
    ----------
    conf_path : str
        Path (absolute or relative) to the TOML configuration file
    overwrite : dict, optional
        A dictionary with values to replace

    Examples
    --------
    ```python
    new_vals = {"base.seed": 10, "model.hidden_dims": 100}
    setup_and_train("/work/conf.toml", overwrite=new_vals)

    Raises
    ------
    RuntimeError
        when no configuration file was found
    """
    log.info("Setting up the experiment...")
    check_conf_path(conf_path)
    conf_ = setup_env_and_get_conf(conf_path=conf_path, overwrite=overwrite)
    Trainer(conf=conf_).prepare().fit()
    return conf_


def setup_and_test(
    conf_path: str = get_default_conf_path(), overwrite: dict | None = None
) -> kit4dl.nn.confmodels.Conf:
    """Set up experiment and run test.

    Parameters
    ----------
    conf_path : str
        Path (absolute or relative) to the TOML configuration file
    overwrite : dict, optional
        A dictionary with values to replace

    Examples
    --------
    ```python
    new_vals = {"base.seed": 10, "model.hidden_dims": 100}
    setup_and_test("/work/conf.toml", overwrite=new_vals)

    Raises
    ------
    RuntimeError
        when no configuration file was found
    """
    log.info("Setting up the experiment...")
    check_conf_path(conf_path)
    conf_ = setup_env_and_get_conf(conf_path=conf_path, overwrite=overwrite)
    Trainer(conf=conf_).prepare().test()
    return conf_


def _overwrite_dict(mapping: dict, overwrite: dict | None = None) -> None:
    # TODO:
    def _replace_recursively(mapping, key, value):
        key_0, *subkeys = key.split(".", maxsplit=1)
        if subkeys:
            return _replace_recursively(
                mapping[key_0], ".".join(subkeys), value
            )
        if key_0 not in mapping:
            return False
        mapping[key_0] = value
        return True

    if not overwrite:
        return
    log.info("Overwriting configuration dictionary with %s", overwrite)
    for key, value in overwrite.items():
        changed = _replace_recursively(mapping, key, value)
        if not changed:
            raise KeyError(
                f"Key {key} not found in the configuration dictionary."
            )


def update_context_from_static() -> None:
    """Update context from static attributes."""
    if not context.VERSION:
        context.VERSION = __version__


def update_context_from_runtime(
    prj_dir: str | None = None,
) -> None:
    """Update context properties from the runtime variables."""
    if not context.PROJECT_DIR:
        context.PROJECT_DIR = prj_dir


def update_context_from_conf(
    conf: kit4dl.nn.confmodels.Conf | None = None,
) -> None:
    """Update context from the configuration file."""
    if not conf:
        return
    if not context.LOG_LEVEL:
        context.LOG_LEVEL = conf.logging.level
    if not context.LOG_FORMAT:
        context.LOG_FORMAT = conf.logging.format_


def _get_conf_dict_from_file(conf_path: str) -> dict:
    assert conf_path and conf_path.endswith(
        ".toml"
    ), "`conf_path` needs to be TOML file!"
    text_load = substitute_symbols(conf_path, **context.get_dict())
    text_load_escaped = escape_os_sep(text_load)
    return toml.loads(text_load_escaped)


def check_conf_path(conf_path: str) -> None:
    """Raise a RuntimeError if conf file does not exist."""
    if not os.path.exists(conf_path):
        log.error("Configuration file %s does not exist", conf_path)
        raise RuntimeError(
            f"the conf file: {conf_path} does not exist. ensure the default"
            " configuration file exist or specify --conf argument to a valid"
            " configuration file."
        )


def setup_env_and_get_conf(
    conf_path: str, overwrite: dict | None = None
) -> kit4dl.nn.confmodels.Conf:
    """Set up environment for the run.

    Parameters
    ----------
    conf_path : str
        A path to the configuration TOML file
    overwrite : dict, optional
        A dictionary to overwrite in the configuration.

    Returns
    -------
    conf : kit4dl.nn.confomodels.Conf
        Configuration object with overwritten key-values
    """
    overwrite = overwrite or {}
    root_dir = os.path.dirname(conf_path)
    prj_dir = os.path.join(os.getcwd(), root_dir)
    sys.path.append(prj_dir)
    update_context_from_static()
    update_context_from_runtime(prj_dir=prj_dir)
    conf_dict = _get_conf_dict_from_file(conf_path)
    _overwrite_dict(conf_dict, overwrite=overwrite)
    conf = kit4dl.nn.confmodels.Conf(**conf_dict, root_dir=root_dir)  # type: ignore[arg-type]
    update_context_from_conf(conf=conf)
    return conf
