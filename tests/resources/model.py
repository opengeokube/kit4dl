"""Exemple module with simple convlution neural network"""
import torch
import torch.nn as nn

from kit4dl import Kit4DLAbstractModule


class SimpleCNN(Kit4DLAbstractModule):
    def configure(self, input_dims, layers, dropout, output_dims):
        self.l1 = nn.Sequential(
            nn.Conv2d(input_dims, 16, kernel_size=3, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        hidden_layers = []
        for _ in range(layers):
            hidden_layers.extend(
                [
                    nn.Conv2d(16, 16, kernel_size=3, bias=True),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                ]
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dims),
        )

    def run_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, label = batch
        x = self.l1(x)
        x = self.hidden_layers(x)
        return label, self.fc(x)
