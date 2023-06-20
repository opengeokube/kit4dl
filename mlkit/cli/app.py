import importlib.resources
import logging
import os
import shutil

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml
import typer
from typing_extensions import Annotated

from mlkit.nn.confmodels import Conf
from mlkit.nn.trainer import Trainer

_app = typer.Typer(name="MLKit")

log = logging.getLogger("MLKit.CLI")


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
    empty_proj_path = importlib.resources.path(
        "mlkit.cli._templates", "project"
    )
    shutil.copytree(empty_proj_path, name)


def _get_conf_from_file(conf, root_dir: str | None = None):
    with open(conf, "rt") as file:
        return Conf(root_dir=root_dir, **toml.load(file))


@_app.command()
def train(
    conf: Annotated[
        str, typer.Option(help="Path to the configuration TOML file")
    ] = None
) -> None:
    """Train using the configuration file

    Parameters
    ----------
    conf : str, optional
        Path to the configuration TOML file.
        If skipped, the program will search for the `conf.toml` file
        in the current working directoy.
    """
    log.info("Attept to run training...")
    if not conf:
        root_dir = os.getcwd()
        log.info(
            (
                "--conf argument was not specified. looking for `conf.toml`"
                " file in the current directory: %s"
            ),
            root_dir,
        )
        conf = os.path.join(root_dir, "conf.toml")
        if not os.path.exists(conf):
            raise RuntimeError(
                "you haven't specified --conf argument and no `conf.toml`"
                f" file was found in the current directory: {root_dir}"
            )
    else:
        root_dir = os.path.dirname(conf)
        if not os.path.exists(conf):
            raise RuntimeError(
                f"the conf file you specified: {conf} does not exist"
            )
    conf_ = _get_conf_from_file(conf, root_dir=root_dir)
    log.info("Running trainer \U0001f3ac")
    Trainer(conf=conf_).prepare().fit()
    log.info("Training finished \U00002728")


def run():
    _app()
