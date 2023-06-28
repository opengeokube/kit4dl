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
from mlkit.nn.confmodels import Conf
from mlkit.nn.trainer import Trainer

_app = typer.Typer(name="MLKit")

log = logging.getLogger("MLKit.CLI")


def update_runtime_context(prj_dir: str, conf: Conf) -> None:
    context.PROJECT_DIR = prj_dir
    context.LOG_LEVEL = conf.base.log_level
    context.LOG_FORMAT = conf.base.log_format


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


def _get_conf_from_file(conf, root_dir: str | None = None):
    with open(conf, "rt", encoding="utf-8") as file:
        return Conf(root_dir=root_dir, **toml.load(file))  # type: ignore[arg-type]


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
    conf_ = _get_conf_from_file(conf, root_dir=root_dir)
    update_runtime_context(prj_dir=prj_dir, conf=conf_)
    log.info("Running trainer \U0001f3ac")
    Trainer(conf=conf_).prepare().fit()
    log.info("Training finished \U00002728")


def run():
    """Run the CLI app."""
    _app()
