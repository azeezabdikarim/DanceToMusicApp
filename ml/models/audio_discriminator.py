import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.nn.modules.batchnorm import BatchNorm1d

class AudioCodeDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_units = 64, num_hidden_layers = 3, alpha = 0.01, sigmoid_out = False):
        super(AudioCodeDiscriminator, self).__init__()
        self.input_size = input_size[0] * input_size[1]
        self.seq_length = int(self.input_size/2)
        self.flatten = nn.Flatten()
        self.codebook_len = 1024
        self.codebook_embed_size = 3
        
        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_units))
        layers.append(nn.LeakyReLU(alpha))
        layers.append(nn.LayerNorm(hidden_units))
        # layers.append(nn.BatchNorm1d(hidden_units))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.LeakyReLU(alpha))
            layers.append(nn.LayerNorm(hidden_units))
            # layers.append(nn.BatchNorm1d(hidden_units))

        self.hidden_layers = nn.Sequential(*layers)

        output_layer = []
        # Output layer
        output_layer.append(nn.Linear(hidden_units, 1))
        if sigmoid_out:
            output_layer.append(nn.Sigmoid())
        self.output_layer = nn.Sequential(*output_layer)


    def forward(self, x):
        # l = self.audio_code_embedding(x[:, :, 0].int())
        # r = self.audio_code_embedding(x[:, :, 1].int())
        # l = self.combine_codebook_embeddings(self.flatten(l))
        # r = self.combine_codebook_embeddings(self.flatten(r))
        # x = torch.cat((l, r), 1)
        x = self.flatten(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class MelSpectrogramDiscriminator(nn.Module):
    def __init__(self, input_shape, hidden_units=64, num_hidden_layers=3, alpha=0.01, sigmoid_out=False):
        super(MelSpectrogramDiscriminator, self).__init__()
        # Define the input shape
        self.channels, self.height, self.width = input_shape
        
        # Initial convolution layer
        layers = [nn.Conv2d(self.channels, hidden_units, kernel_size=3, stride=1, padding=1),
                  nn.LeakyReLU(alpha),
                  nn.InstanceNorm2d(hidden_units)]
        
        # Hidden convolution layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(alpha))
            layers.append(nn.InstanceNorm2d(hidden_units))
        
        # Adaptive pooling layer to make the output of conv layers a fixed size
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units, 1),
        )
        
        # Optional sigmoid activation
        if sigmoid_out:
            self.output_layer.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x
