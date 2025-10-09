import torch as t
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, x_dim=2, y_dim=2, hidden_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, x):
        return self.model(x)

