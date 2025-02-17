import torch
import torch.nn as nn
import torch.nn.init as init


class SequenceRegressor(nn.Module):

    def __init__(
        self,
        hidden_dim,
        output_dim=1,
        dropout_rate=0.0,
        dtype=torch.float64,
        device: str = "cpu",
        bias=True,
    ):
        super().__init__()

        self.dense = nn.Linear(
            hidden_dim, hidden_dim, bias=bias, dtype=dtype, device=device
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

        self.out_proj = nn.Linear(
            hidden_dim, output_dim, bias=bias, dtype=dtype, device=device
        )
    
        self.init_weights()

    def forward(self, features):
        x = self.dropout(features[:, 0, :])
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                # Initialize weights
                init.xavier_uniform_(param)
            elif "bias" in name:
                # Initialize biases
                init.constant_(param, 0.0)
            else:
                # Fallback for unhandled parameters
                raise ValueError(f"Unhandled parameter type for {name}")