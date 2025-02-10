import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 3,
        dropout_rate: float = 0.0,
        eps: float = 1e-6,
        concat: bool = False,
        dtype: torch.dtype = torch.float64,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        device: str = "cpu",
    ):

        super(FFN, self).__init__()

        self.eps = eps
        self.hidden_dim = hidden_dim
        self.concat_dim = hidden_dim * 2
        self.concat = concat
        self.depth = depth
        self.activation = activation
        self.device = device

        # Define linear layers
        input_dim = self.concat_dim if concat else hidden_dim
        self.dense = nn.ModuleList([
            nn.Linear(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                bias=bias,
                dtype=dtype,
                device=device,
            )
            for i in range(depth)
        ])

        # Define dropout layers
        self.dropout = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(depth)])

        self.init_weight()

    def forward(self, u: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:

        # Validate inputs
        if self.concat:
            # Concatenate inputs
            combined = torch.cat([u, v], dim=-1)
        else:
            combined = u

        # Apply layers
        x = combined
        for i in range(self.depth):
            x = self.dropout[i](x)
            x = self.dense[i](x)
            x = self.activation(x)

        return x

    def init_weight(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                # Initialize weights
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
            elif "bias" in name:
                # Initialize biases
                nn.init.constant_(param, 0.0)
            else:
                # Fallback for unhandled parameters
                raise ValueError(f"Unhandled parameter type for {name}")