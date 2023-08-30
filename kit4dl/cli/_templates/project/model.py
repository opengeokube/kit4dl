"""Docstring of the module containing my neural network module."""
from typing import Any

import torch

from kit4dl import Kit4DLAbstractModule


class MyNewNetwork(Kit4DLAbstractModule):
    """My network module."""

    def configure(self, *args: Any, **kwargs: Any) -> None:
        """Configure the architecture."""
        # You can define kwargs used in the configuration file
        # def setup(self, input_dims, dropout_rate):
        #     ...
        # Here, you define the architecture
        raise NotImplementedError

    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Run single step."""
        raise NotImplementedError
