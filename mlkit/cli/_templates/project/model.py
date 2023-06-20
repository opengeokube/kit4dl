"""Docstring of the module containing my neural network module"""
import torch

from mlkit import MLKitAbstractModule


class MyNewNetwork(MLKitAbstractModule):
    def setup(self, **kwargs) -> None:
        # You can define kwargs used in the configuration file
        # def setup(self, input_dims, dropout_rate):
        #     ...
        # Here, you define the architecture
        raise NotImplementedError

    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
