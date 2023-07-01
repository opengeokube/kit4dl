"""Module with CLI for MLKit."""
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

from mlkit import context
from mlkit.formatting import escape_os_sep, substitute_symbols
from mlkit.nn.confmodels import Conf
from mlkit.nn.trainer import Trainer

_app = typer.Typer(name="MLKit")

log = logging.getLogger("MLKit.CLI")


def update_context_from_runtime(
    prj_dir: str | None = None,
) -> None:
    """Update context properties."""
    context.PROJECT_DIR = prj_dir  # type: ignore[attr-defined]


def update_context_from_conf(conf: Conf | None = None) -> None:
    if not conf:
        return
    context.LOG_LEVEL = conf.base.log_level  # type: ignore[attr-defined]
    context.LOG_FORMAT = conf.base.log_format  # type: ignore[attr-defined]


@_app.command()
def init(
    name: Annotated[
        str, typer.Option(help="The name of your new project")
    ] = "new_mlkit_project"
) -> None:
    """Create a new MLKit project.

    Parameters
    ----------
    name : str, optional
        The optional name of the project.
        If skipped, the deafult `new_mlkit_project` will be used
    """
    log.info("MLKit Creating a new skeleton for the project: << %s >>", name)
    with importlib.resources.path(
        "mlkit.cli._templates", "project"
    ) as empty_proj_path:
        shutil.copytree(empty_proj_path, name)


def _get_conf_from_file(conf_path: str, root_dir: str | None = None):
    assert conf_path and conf_path.endswith(
        ".toml"
    ), "`conf_path` needs to be TOML file!"
    with open(conf_path, "rt", encoding="utf-8") as file:
        text_load = substitute_symbols(file.read(), **context.get_dict())
        text_load_escaped = escape_os_sep(text_load)
        return Conf(root_dir=root_dir, **toml.loads(text_load_escaped))  # type: ignore[arg-type]


def _get_default_conf_path() -> str:
    return os.path.join(os.getcwd(), "conf.toml")


@_app.command()
def train(
    conf: Annotated[
        str, typer.Option(help="Path to the configuration TOML file")
    ] = _get_default_conf_path()
) -> None:
    """Train using the configuration file.

    Parameters
    ----------
    conf : str, optional
        Path to the configuration TOML file.
        If skipped, the program will search for the `conf.toml` file
        in the current working directoy.
    """
    log.info("Attept to run training...")
    root_dir = os.path.dirname(conf)
    if not os.path.exists(conf):
        raise RuntimeError(
            f"the conf file: {conf} does not exist. ensure the default"
            " configuration file exist or specify --conf argument to a valid"
            " configuration file."
        )
    prj_dir = os.path.join(os.getcwd(), root_dir)
    sys.path.append(prj_dir)
    update_context_from_runtime(prj_dir=prj_dir)
    conf_ = _get_conf_from_file(conf, root_dir=root_dir)
    update_context_from_conf(conf=conf_)
    log.info("Running trainer \U0001f3ac")
    Trainer(conf=conf_).prepare().fit()
    log.info("Training finished \U00002728")


def run():
    """Run the CLI app."""
    _app()
