import math
import torch
import torch.nn as nn
from opt_einsum import contract
import os

class MultiHeadStrassenAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        dropout_rate=0.0,
        dtype=torch.float64,
        bias=False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = False,
        max_positions: int = 128,
        masked_multihead_attention: bool = False,
        is_cross_attention: bool = False,
    ):
        super(MultiHeadStrassenAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.max_positions = max_positions
        self.masked_multihead_attention = masked_multihead_attention
        self.is_cross_attention = is_cross_attention
        
        assert (
            self.hidden_dim % num_heads == 0
        ), "hidden_dim must be divisible by num_heads."


        if self.is_cross_attention:
            self.A  = nn.Linear(
                self.hidden_dim, self.hidden_dim * 1, bias=bias, dtype=dtype, device=device
            )
            self.BCV1V2 = nn.Linear(
                self.hidden_dim, self.hidden_dim * 4, bias=bias, dtype=dtype, device=device
            )
        else:
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

    def forward(self,
                query_hidden_states, #decoder hidden states
                key_value_hidden_states=None, # encoder hidden states
                batch_mask=None):
        
        src_key_padding_mask, query_padding_mask = None, None 
        if batch_mask is not None and self.is_cross_attention:
            src_key_padding_mask, query_padding_mask = batch_mask["key_padding_mask"], batch_mask["query_padding_mask"] 
            key_padding_mask = self.construct_mask(src_key_padding_mask)
            query_padding_mask = self.construct_mask(query_padding_mask)
        
        B, N, C = query_hidden_states.shape
        H = self.num_heads
        D = self.head_dim

        if self.is_cross_attention:
            qx = self.A(query_hidden_states).reshape(B, N, 1, H, D).permute(2, 0, 3, 1, 4)
            cx = self.ABV1V2(key_value_hidden_states).reshape(B, N, 4, H, D).permute(2, 0, 3, 1, 4)
            a, b, c, vj, vk = qx[0], cx[0], cx[1], cx[2], cx[3]
        else:
            wx = self.w(query_hidden_states).reshape(B, N, 5, H, D).permute(2, 0, 3, 1, 4)
            a, b, c, vj, vk = wx[0], wx[1], wx[2], wx[3], wx[4]

        X = contract("bhjd,bhkd->bhjk", b, c) * self.scaler
        Y = contract("bhkd,bhid->bhki", c, a) * self.scaler
        Z = contract("bhid,bhjd->bhij", a, b) * self.scaler

        if query_padding_mask is not None: #query attention mask
            Z = Z.masked_fill(query_padding_mask, self.mask_padding_value)
            Y = Y.masked_fill(query_padding_mask, self.mask_padding_value)
        if src_key_padding_mask is not None: #key attention mask
            X = X.masked_fill(key_padding_mask, self.mask_padding_value)
            Y = Y.masked_fill(key_padding_mask, self.mask_padding_value)
            Z = Z.masked_fill(key_padding_mask, self.mask_padding_value)
        if self.masked_multihead_attention: #causal mask
            X = X.masked_fill(self.tril[:N,:N], self.mask_padding_value)
            Y = Y.masked_fill(self.tril[:N,:N], self.mask_padding_value)
            Z = Z.masked_fill(self.tril[:N,:N], self.mask_padding_value)
        
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