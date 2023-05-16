"""A module with train task definition"""
import argparse
import os

import toml


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MLKit",
        description="A quick way to start with machine and deep learning",
    )
    parser.add_argument(
        "--conf",
        default="conf.toml",
        help="TOML file with neural network configuration",
        type=str,
    )
    return parser.parse_args()


def run() -> None:
    from mlkit.nn.confmodels import Conf
    from mlkit.nn.trainer import Trainer

    arguments = parse_args()
    breakpoint()
    assert os.path.exists(
        arguments.conf
    ), f"configuration file `{arguments.conf}` does not exist!"
    with open(arguments.conf, "rt") as conf_file:
        conf = Conf(**toml.load(conf_file))
    Trainer(conf=conf).prepare().fit()


if __name__ == "__main__":
    run()
