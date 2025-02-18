import torch
import torch.nn as nn

from models.transformers.attentions.backbone_attention import BackboneAttention
from models.transformers.feed_forward_networks.attention import FFN

class BackboneTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str = "standard",
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        use_norm: bool = True,
        use_attention_dropout: bool = False,
        dtype: torch.dtype = torch.float64,
        concat: bool = False,
        eps: float = 1e-6,
        ffn_depth: int = 3,
        device: str = "cpu",
        is_edge: bool = False,
    ):

        super(BackboneTransformerLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.concat = concat
        self.eps = eps
        self.attention_type = attention_type
        self.is_edge = is_edge

        # Attention mechanism
        self.attention = BackboneAttention(
            attention_type=attention_type,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_dropout=use_attention_dropout,
            dropout_rate=dropout_rate,
            dtype=dtype,
            device=device,
            is_edge=is_edge,
        )

        # Feed-forward network
        self.ffn = FFN(
            hidden_dim=hidden_dim,
            depth=ffn_depth,
            dropout_rate=dropout_rate,
            eps=eps,
            concat=concat,
            dtype=dtype,
            device=device,
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(
        self, hidden_state: torch.Tensor, batch_mask: torch.Tensor = None
    ) -> torch.Tensor:

        # Attention block with residual connection
        hidden_state = self.norm1(hidden_state) if self.use_norm else hidden_state
        attention_output, attention_weights = self.attention(
            hidden_state=hidden_state, batch_mask=batch_mask
        )
        attention_output = self.dropout1(attention_output)

        if self.concat:
            # Concatenate attention output with the hidden state: x <- x + f(x, dt*theta)
            if self.use_norm:
                attention_output = self.norm2(attention_output)
            ffn_output = self.ffn(u=hidden_state, v=attention_output)
            hidden_state = hidden_state + self.dropout2(ffn_output)
        else:
            # Add attention output to the hidden state: x <- x + dt*theta + f(ddt*(x+theta))
            hidden_state = hidden_state + attention_output
            if self.use_norm:
                hidden_state = self.norm2(hidden_state)
            ffn_output = self.ffn(u=hidden_state)
            hidden_state = hidden_state + self.dropout2(ffn_output)

        return hidden_state, attention_weights