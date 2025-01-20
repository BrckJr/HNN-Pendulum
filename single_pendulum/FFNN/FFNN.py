import torch
import torch.nn as nn

class FFNN(nn.Module):
    """ A simple Feedforward Neural Network (FFNN) with two hidden layers """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the network """
        return self.model(x)