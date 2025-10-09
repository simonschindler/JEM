import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with one hidden layer.

    This model can be used as a general-purpose function approximator.
    """

    def __init__(self, x_dim: int = 2, y_dim: int = 2, hidden_dim: int = 128):
        """
        Initializes the MLP.

        Args:
            x_dim (int): The dimensionality of the input features.
            y_dim (int): The dimensionality of the output.
            hidden_dim (int): The number of neurons in the hidden layer.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            The output tensor.
        """
        return self.model(x)
