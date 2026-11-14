"""Shared MLP with state_dict keys compatible with ServiceFlow checkpoints."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.1):
        super().__init__()
        layers = []
        previous = input_size
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)])
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)
