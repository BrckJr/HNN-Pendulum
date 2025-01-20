import torch
import torch.nn as nn

class HNN(nn.Module):
    """ Hamiltonian Neural Network model """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(HNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the network """
        return self.model(x)

    def initialize_weights(self):
        """ Initialize weights using Xavier Initialization """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)