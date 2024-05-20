"""Module with CLI for Kit4DL."""

import importlib.resources
import logging
import os
import shutil
from typing import Optional


import typer
from typing_extensions import Annotated

from kit4dl import (
    _version,
    setup_env_and_get_conf,
    get_default_conf_path,
    configure_logger,
)
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

log = logging.getLogger("Kit4DL.CLI")
configure_logger(log, _CLI_LOG_LEVEL, _CLI_LOG_FORMAT)


# ##############################
#         UTILS METHODS
# ##############################
def _remove_redundant_items(path):
    shutil.rmtree(os.path.join(path, "__pycache__"), ignore_errors=True)


def parse_overwriting_options(
    ctx: typer.Context, value: str | None
) -> dict[str, str]:
    """Parse extra arguments for the `overwrite` option."""
    assert ctx
    custom_dict: dict = {}
    if not value:
        return custom_dict
    for kv_pair in value.split(","):
        try:
            key, value = kv_pair.split("=")
        except ValueError as err:
            raise typer.BadParameter(
                f"Invalid key-value pair: {kv_pair}"
            ) from err
        custom_dict[key] = value
    return custom_dict


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
    _template_pkg = "kit4dl.cli._templates"
    log.info("Kit4DL Creating a new skeleton for the project: << %s >>", name)
    assert not os.path.exists(name), f"The project `{name}` already exists!"
    with importlib.resources.as_file(
        importlib.resources.files(_template_pkg).joinpath("project")
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
def test(
    conf: Annotated[
        str, typer.Option(help="Path to the configuration TOML file")
    ] = get_default_conf_path(),
):
    """Test using the configuration file.

    Parameters
    ----------
    conf : str, optional
        Path to the configuration TOML file.
        If skipped, the program will search for the `conf.toml` file
        in the current working directory.
    """
    conf_ = setup_env_and_get_conf(conf_path=conf)
    log.info("Attempt to run testing...")
    assert (
        conf_.training.checkpoint_path
    ), "`checkpoint_path` is not defined, i need to load model state."
    trainer = Trainer(conf=conf_).prepare()
    log.info("Running testing \U00002728")
    trainer.test()
    log.info("Testing finished \U00002728")


@_app.command()
def train(
    conf: Annotated[
        str, typer.Option(help="Path to the configuration TOML file")
    ] = get_default_conf_path(),
    skiptest: Annotated[
        bool,
        typer.Option(help="If testing (using best weights) should be skipped"),
    ] = False,
    overwrite: Annotated[
        Optional[str],
        typer.Option(
            ...,
            callback=parse_overwriting_options,
            help="Comma-separated key-value pairs (KEY=VALUE)",
        ),
    ] = None,
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
    conf_ = setup_env_and_get_conf(conf_path=conf, overwrite=overwrite)  # type: ignore[arg-type]
    log.info("Running trainer \U0001f3ac")
    trainer = Trainer(conf=conf_).prepare()
    trainer.fit()
    log.info("Training finished \U00002728")
    if not skiptest:
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
