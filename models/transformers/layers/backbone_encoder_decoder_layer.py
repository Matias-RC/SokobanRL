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
        block_size: int = 128,
        masked_multihead_attention: bool = False,
        is_cross_attention: bool = False,
    ):

        super(BackboneTransformerLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.concat = concat
        self.eps = eps
        self.attention_type = attention_type
        self.masked_multihead_attention = masked_multihead_attention
        self.is_cross_attention = is_cross_attention
        
        # 1- Attention mechanism for multihead attention
        self.self_attention = BackboneAttention(
            attention_type=attention_type,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_dropout=use_attention_dropout,
            dropout_rate=dropout_rate,
            dtype=dtype,
            device=device,
            block_size=block_size,
            masked_multihead_attention=masked_multihead_attention,
            is_cross_attention=False,
        )
        # Feed-forward network for multihead attention
        self.ffn_mha = FFN(
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

        if self.is_cross_attention:
            # Attention mechanism for cross multi-head attention
            self.cross_attention = BackboneAttention(attention_type=attention_type,
                                                     hidden_dim=hidden_dim,
                                                     num_heads=num_heads,
                                                     use_dropout=use_attention_dropout,
                                                     dropout_rate=dropout_rate,
                                                     dtype=dtype,
                                                     device=device,
                                                     block_size=block_size,
                                                     masked_multihead_attention=False,
                                                     is_cross_attention=is_cross_attention,)
            
            # Feed-forward network for cross multi-head attention
            self.ffn_cmha = FFN(
                hidden_dim=hidden_dim,
                depth=ffn_depth,
                dropout_rate=dropout_rate,
                eps=eps,
                concat=concat,
                dtype=dtype,
                device=device,
            )
            #normalization and dropout for cross attention
            self.norm3 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
            self.dropout3 = nn.Dropout(dropout_rate)
            self.dropout4 = nn.Dropout(dropout_rate)

    def forward(
        self, query_hidden_states: torch.Tensor,
        cross_hidden_states: torch.Tensor=None,
        batch_mask: torch.Tensor=None, #self, hidden_state: torch.Tensor, batch_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        print(self.masked_multihead_attention)
        print(self.is_cross_attention)
        print(query_hidden_states.shape)
        print(self.hidden_dim)
        # Step 1: Multihead Attention 
        hidden_state = self.norm1(query_hidden_states) if self.use_norm else hidden_state
        print("hidden state")
        
        attention_output, attention_weights = self.self_attention(
            query_hidden_states, cross_hidden_states=cross_hidden_states, batch_mask=batch_mask
        )
        print("attention")
        
        attention_output = self.dropout1(attention_output)
        print("concat")
        if self.concat:
            # Concatenate attention output with the hidden state: x <- x + f(x, dt*theta)
            if self.use_norm:
                attention_output = self.norm2(attention_output)
            ffn_output = self.ffn_mha(u=hidden_state, v=attention_output)
            hidden_state = hidden_state + self.dropout2(ffn_output)
        else:
            # Add attention output to the hidden state: x <- x + dt*theta + f(ddt*(x+theta))
            hidden_state = hidden_state + attention_output
            if self.use_norm:
                hidden_state = self.norm2(hidden_state)
            ffn_output = self.ffn_mha(u=hidden_state)
            hidden_state = hidden_state + self.dropout2(ffn_output)

        # Step 2: Cross Multihead Attention
        if self.is_cross_attention:
            attention_output, attention_weights = self.cross_attention(
                query_hidden_states, cross_hidden_states=cross_hidden_states, batch_mask=batch_mask
            )
            attention_output = self.dropout3(attention_output)
            if self.concat:
                # Concatenate attention output with the hidden state: x <- x + f(x, dt*theta)
                if self.use_norm:
                    attention_output = self.norm3(attention_output)
                ffn_output = self.ffn_cmha(u=hidden_state, v=attention_output)
                hidden_state = hidden_state + self.dropout4(ffn_output)
            else:
                # Add attention output to the hidden state: x <- x + dt*theta + f(ddt*(x+theta))
                hidden_state = hidden_state + attention_output
                if self.use_norm:
                    hidden_state = self.norm3(hidden_state)
                ffn_output = self.ffn_cmha(u=hidden_state)
                hidden_state = hidden_state + self.dropout4(ffn_output)

        return hidden_state, attention_weights