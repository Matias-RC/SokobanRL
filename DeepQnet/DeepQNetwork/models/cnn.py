import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros((1, input_channels, 10, 10))  # Assuming 10x10 input
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.fc = nn.Linear(cnn_out_dim, output_dim)
    
    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)