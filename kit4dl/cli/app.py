"""Module with CLI for Kit4DL."""
import importlib.resources
import logging
import os
import shutil
import sys

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml  # type: ignore[no-redef]

import typer
from typing_extensions import Annotated

from kit4dl import _version, context
from kit4dl.formatting import escape_os_sep, substitute_symbols
from kit4dl.nn.confmodels import Conf
from kit4dl.nn.trainer import Trainer

# ##############################
#         CREATE CLI
# ##############################
_app = typer.Typer(name="Kit4DL")

# ##############################
#       CONFIGURE LOGGER
# ##############################
_CLI_LOG_LEVEL = "INFO"
_CLI_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _configure_logger(logger: logging.Logger):
    logger.setLevel(_CLI_LOG_LEVEL)
    formatter = logging.Formatter(_CLI_LOG_FORMAT)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    for hdl in log.handlers:
        hdl.setFormatter(formatter)
        hdl.setLevel(_CLI_LOG_LEVEL)  # type: ignore[arg-type]


log = logging.getLogger("Kit4DL.CLI")
_configure_logger(log)


# ##############################
#         UTILS METHODS
# ##############################
def update_context_from_static() -> None:
    """Update context from static attributes."""
    context.VERSION = _version.__version__


def update_context_from_runtime(
    prj_dir: str | None = None,
) -> None:
    """Update context properties from the runtime variables."""
    context.PROJECT_DIR = prj_dir


def update_context_from_conf(conf: Conf | None = None) -> None:
    """Update context from the configuration file."""
    if not conf:
        return
    context.LOG_LEVEL = conf.logging.level
    context.LOG_FORMAT = conf.logging.format_


def _get_conf_from_file(conf_path: str, root_dir: str | None = None):
    assert conf_path and conf_path.endswith(
        ".toml"
    ), "`conf_path` needs to be TOML file!"
    text_load = substitute_symbols(conf_path, **context.get_dict())
    text_load_escaped = escape_os_sep(text_load)
    return Conf(root_dir=root_dir, **toml.loads(text_load_escaped))  # type: ignore[arg-type]


def _get_default_conf_path() -> str:
    return os.path.join(os.getcwd(), "conf.toml")


def _remove_redundant_items(path):
    shutil.rmtree(os.path.join(path, "__pycache__"), ignore_errors=True)


def _is_test_allowed(trainer: Trainer) -> bool:
    return trainer.is_finished


# ##############################
#      COMMANDS DEFINITIONS
# ##############################
@_app.command()
def init(
    name: Annotated[
        str, typer.Option(help="The name of your new project")
    ] = "new_kit4dl_project"
) -> None:
    """Create a new Kit4DL project.

    Parameters
    ----------
    name : str, optional
        The optional name of the project.
        If skipped, the deafult `new_kit4dl_project` will be used
    """
    log.info("Kit4DL Creating a new skeleton for the project: << %s >>", name)
    assert not os.path.exists(name), f"The project `{name}` already exists!"
    with importlib.resources.path(
        "kit4dl.cli._templates", "project"
    ) as empty_proj_path:
        shutil.copytree(empty_proj_path, name)
    _remove_redundant_items(name)


@_app.command()
def resume(
    checkpoint: Annotated[
        str, typer.Option(help="Path to the checkpoint file")
    ]
):
    """Resume learning from the checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to a checkpoint file
    """
    raise NotImplementedError


@_app.command()
def train(
    conf: Annotated[
        str, typer.Option(help="Path to the configuration TOML file")
    ] = _get_default_conf_path(),
    test: Annotated[
        bool, typer.Option(help="If perform testing using best weights")
    ] = True,
) -> None:
    """Train using the configuration file.

    Parameters
    ----------
    conf : str, optional
        Path to the configuration TOML file.
        If skipped, the program will search for the `conf.toml` file
        in the current working directory.
    test
    """
    log.info("Attempt to run training...")
    root_dir = os.path.dirname(conf)
    if not os.path.exists(conf):
        raise RuntimeError(
            f"the conf file: {conf} does not exist. ensure the default"
            " configuration file exist or specify --conf argument to a valid"
            " configuration file."
        )
    prj_dir = os.path.join(os.getcwd(), root_dir)
    sys.path.append(prj_dir)
    update_context_from_static()
    update_context_from_runtime(prj_dir=prj_dir)
    conf_ = _get_conf_from_file(conf, root_dir=root_dir)
    update_context_from_conf(conf=conf_)
    log.info("Running trainer \U0001f3ac")
    trainer = Trainer(conf=conf_).prepare()
    trainer.fit()
    log.info("Training finished \U00002728")
    if test:
        if not _is_test_allowed(trainer):
            return
        log.info("Running testing \U00002728")
        trainer.test()
        log.info("Testing finished \U00002728")


@_app.command()
def version() -> None:
    """Display Kit4DL version."""
    print(_version.__version__)


def run():
    """Run the CLI app."""
    _app()
