import math
import torch
import torch.nn as nn
from opt_einsum import contract
import os

class StrassenAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        dropout_rate=0.0,
        dtype=torch.float64,
        bias=False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = False
    ):
        super(StrassenAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device
        self.dtype = dtype
        self.bias = bias

        assert (
            self.hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads."

        self.w = nn.Linear(
            self.hidden_dim, self.hidden_dim * 5, bias=bias, dtype=dtype, device=device
        )
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
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

        wx = self.w(hidden_state).reshape(B, N, 5, H, D).permute(2, 0, 3, 1, 4)
        a, b, c, vj, vk = wx[0], wx[1], wx[2], wx[3], wx[4]

        X = contract("bhjd,bhkd->bhjk", b, c) * self.scaler
        Y = contract("bhkd,bhid->bhki", c, a) * self.scaler
        Z = contract("bhid,bhjd->bhij", a, b) * self.scaler

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

            X = X.masked_fill(mask_bc.bool(), self.mask_padding_value)
            Y = Y.masked_fill(mask_ca.bool(), self.mask_padding_value)
            Z = Z.masked_fill(mask_ab.bool(), self.mask_padding_value)

        X = (
            X
            - torch.max(
                torch.max(X, dim=-1, keepdim=True).values, dim=-2, keepdim=True
            ).values
        )
        Y = Y - torch.max(Y, dim=-2, keepdim=True).values
        Z = Z - torch.max(Z, dim=-1, keepdim=True).values

        X = X.exp()
        Y = Y.exp()
        Z = Z.exp()

        if self.use_dropout:
            X = self.dropout(X)
            Y = self.dropout(Y)
            Z = self.dropout(Z)

        X_vj = contract("bhjk,bhjd->bhjkd", X, vj)
        Y_vk = contract("bhki,bhkd->bhikd", Y, vk)
        ZX_vj = contract("bhij,bhjkd->bhikd", Z, X_vj)
        Num = contract("bhikd,bhikd->bhid", Y_vk, ZX_vj)

        YX_T = contract("bhki,bhjk->bhij", Y, X)
        Den = contract("bhij,bhij->bhi", Z, YX_T)
        Den = Den + 1e-9

        att = Num / Den.unsqueeze(-1)

        att = att.transpose(1, 2)
        att = att.reshape(B, N, C)

        return att, None

    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)  # Ensure directory exists

        torch.save(self.w.weight.data.cpu(), os.path.join(path, "w.pt"))

        if self.bias:
            torch.save(self.w.bias.data.cpu(), os.path.join(path, "w_bias.pt"))