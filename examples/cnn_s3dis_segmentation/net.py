"""Exemple module with simple convlution neural network"""
import torch
import torch.nn as nn

from mlkit import MLKitAbstractModule


class SimpleSegmentationNetwork(MLKitAbstractModule):
    def configure(self, input_dims, output_dims) -> None:
        self.l1 = nn.Sequential(
            nn.Conv1d(
                input_dims, 128, kernel_size=3, padding="same", bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(
                128, output_dims, kernel_size=3, padding="same", bias=True
            ),
        )

    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, cat, instance = batch
        res = self.l1(x.permute(0, 2, 1))
        return cat, res
