import math
import torch
import torch.nn as nn
from opt_einsum import contract


class StrassenAttentionWithALiBi(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        dropout_rate=0.0,
        dtype=torch.float64,
        bias=False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = True,
    ):
        super(StrassenAttentionWithALiBi, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device
        self.dtype = dtype

        assert (
            self.hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads."

        self.w = nn.Linear(
            self.hidden_dim, self.hidden_dim * 5, bias=bias, dtype=dtype, device=device
        )
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Precompute ALiBi slopes
        self.alibi_slopes = self.compute_alibi_slopes(num_heads).to(device)


    def compute_alibi_slopes(self, num_heads):
        """
        Generate slopes for ALiBi using the geometric progression formula:
        - Start: \( 2^{-\frac{8}{n}} \)
        - Ratio: \( 2^{-\frac{8}{n}} \)
        """
        base = 2 ** (-8 / num_heads)
        slopes = [base ** i for i in range(1, num_heads + 1)]
        return torch.tensor(slopes, dtype=torch.float32)

    def construct_mask(self, attention_mask):
        attention_mask = attention_mask.flatten(1)
        expanded_mask = attention_mask.unsqueeze(1) + attention_mask.unsqueeze(2)
        expanded_mask = expanded_mask.unsqueeze(1)
        expanded_mask = expanded_mask.to(self.device)
        return expanded_mask
    
    def forward(self, hidden_state, batch_mask=None):

        attention_mask, mask = None, None
        if batch_mask is not None:
            attention_mask, mask = batch_mask["attention_mask"], batch_mask["mask"]

        B, N, C = hidden_state.shape
        H = self.num_heads
        D = self.head_dim

        w = self.w(hidden_state).reshape(B, N, 5, H, D).permute(2, 0, 3, 1, 4)
        a, b, c, v1, v2 = w[0], w[1], w[2], w[3], w[4]

        X = contract("bhid,bhjd->bhij", a, b) * self.scaler
        Y = contract("bhjd,bhkd->bhjk", b, c) * self.scaler
        Z = contract("bhkd,bhid->bhki", c, a) * self.scaler

        #use ALiBi relative positional encoding
        position_bias = torch.arange(N, device=self.device).unsqueeze(0) - torch.arange(N, device=self.device).unsqueeze(1)
        position_bias_rev = torch.arange(N, device=self.device).unsqueeze(1) - torch.arange(N, device=self.device).unsqueeze(0)
        position_bias[position_bias > 0] = 0
        position_bias_rev[position_bias_rev > 0] = 0
        position_bias_encoder = position_bias + position_bias_rev
        position_bias_encoder = position_bias_encoder.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N, N)
        position_bias_encoder = position_bias_encoder * self.alibi_slopes.view(1, H, 1, 1)  # Broadcast slopes to heads
        X += position_bias
        Y += position_bias
        Z += position_bias

        if attention_mask is not None or mask is not None:
            if attention_mask is not None:
                expanded_mask = self.construct_mask(attention_mask)
                
                mask_bc = expanded_mask if mask is None else expanded_mask + mask["BC"].unsqueeze(1).to(self.device)
                mask_ca = expanded_mask if mask is None else expanded_mask + mask["CA"].unsqueeze(1).to(self.device)
                mask_ab = expanded_mask if mask is None else expanded_mask + mask["AB"].unsqueeze(1).to(self.device)
            else:
                mask_bc = mask["BC"].unsqueeze(1).to(self.device)
                mask_ca = mask["CA"].unsqueeze(1).to(self.device)
                mask_ab = mask["AB"].unsqueeze(1).to(self.device)

            X = X.masked_fill(mask_ab, self.mask_padding_value)
            Y = Y.masked_fill(mask_bc, self.mask_padding_value)
            Z = Z.masked_fill(mask_ca, self.mask_padding_value)

        X = X - torch.max(X, dim=-1, keepdim=True).values
        Y = (
            Y
            - torch.max(
                torch.max(Y, dim=-1, keepdim=True).values, dim=-2, keepdim=True
            ).values
        )
        Z = Z - torch.max(Z, dim=-2, keepdim=True).values

        X = X.exp()
        Y = Y.exp()
        Z = Z.exp()

        if self.use_dropout:
            X = self.dropout(X)
            Y = self.dropout(Y)
            Z = self.dropout(Z)
        
        V = contract("bhjd,bhkd->bhjkd", v1, v2)

        up = contract("bhikd,bhki->bhid", contract("bhij,bhjk,bhjkd->bhikd", X, Y, V), Z)
        down = contract("bhik,bhki->bhi", contract("bhij,bhjk->bhik", X, Y), Z)
        down = down + 1e-9

        att = up / down.unsqueeze(-1)
        
        att = att.transpose(1, 2)
        att = att.reshape(B, N, C)

        return att, None