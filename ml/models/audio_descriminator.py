import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.nn.modules.batchnorm import BatchNorm1d

class AudioDescriminator(nn.Module):
    def __init__(self, input_size, hidden_units = 64, num_hidden_layers = 3, alpha = 0.01, sigmoid_out = False):
        super(AudioDescriminator, self).__init__()
        self.input_size = input_size[0] * input_size[1]
        self.flatten = nn.Flatten()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_units))
        layers.append(nn.LeakyReLU(alpha))
        # layers.append(nn.BatchNorm1d(hidden_units))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.LeakyReLU(alpha))
            # layers.append(nn.BatchNorm1d(hidden_units))

        self.hidden_layers = nn.Sequential(*layers)

        output_layer = []
        # Output layer
        output_layer.append(nn.Linear(hidden_units, 1))
        if sigmoid_out:
            output_layer.append(nn.Sigmoid())
        self.output_layer = nn.Sequential(*output_layer)


    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
