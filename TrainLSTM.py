import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from Logic import master
import SokoSource as source


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        # If hidden and cell states are not provided, initialize them as zeros
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        return out, hn, cn


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_hidden_layers):
        super(MLP, self).__init__()
        
        # List to hold layers
        layers = []
        
        # First hidden layer (input to first hidden layer)
        layers.append(nn.Linear(in_dim, hid_dim))
        layers.append(nn.ReLU())
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hid_dim, out_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    

def generateSequence(length):
    return [random.randint(0, 3) for _ in range(length)]

def ConvertSeqToOneHot(seq):
    new_seq = []
    for i in seq:
        if i == 0:
            new_seq.append(1,0,0,0)
        elif i == 1:
            new_seq.append(0,1,0,0)
        elif i == 2:
            new_seq.append(0,0,1,0)
        else:
            new_seq.append(0,0,0,1)

def train(Samples, encoder, decoder, lr):
    return

