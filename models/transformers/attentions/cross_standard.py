import torch
import torch.nn as nn
import math
from opt_einsum import contract
import os

class CrossStandardAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        bias: bool = False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = True,
        is_cross_attention = True,

    ):
        super(CrossStandardAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaler = 1 / math.sqrt(self.head_dim)
        self.mask_padding_value = mask_padding_value
        self.device = device
        self.bias = bias
        self.is_cross_attention = is_cross_attention

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        #query, key and value matrices
        self.Q = nn.Linear(self.hidden_dim, self.hidden_dim * 1, bias=bias, dtype=dtype, device=device)
        self.KV = nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=bias, dtype=dtype, device=device)

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
    def construct_mask(self, attention_mask):
        attention_mask = attention_mask.flatten(1)
        expanded_mask = attention_mask.unsqueeze(1) + attention_mask.unsqueeze(2)
        expanded_mask = expanded_mask.unsqueeze(1)     
        expanded_mask = expanded_mask.repeat(1, self.num_heads, 1, 1)
        expanded_mask = expanded_mask.to(self.device)
        return expanded_mask
   
    def forward(self,hidden_state,memory=None,batch_mask=None):

        
        attention_mask, mask = None, None
        if batch_mask is not None:
            attention_mask, mask, tgt_mask = batch_mask["attention_mask"], batch_mask["mask"], batch_mask["tgt_mask"]

        B, N, C = hidden_state.shape
        H = self.num_heads
        D = self.head_dim
        assert (
            C == self.hidden_dim
        ), "Last dimension of hidden_state must match hidden_dim."

        Qx = self.Q(hidden_state).reshape(B, N, 1, H, D).permute(2, 0, 3, 1, 4)
        KVx = self.KV(memory).reshape(B, N, 2, H, D).permute(2, 0, 3, 1, 4)

        q, k, v = Qx[0], KVx[0], KVx[1]

        scores = contract("bhid,bhjd->bhij", q, k) * self.scaler

        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask)
            scores = scores.masked_fill(expanded_mask, self.mask_padding_value)
        
        if tgt_mask is not None:
            tgt_expanded_mask = self.construct_mask(tgt_mask)
            scores = scores.masked_fill(tgt_expanded_mask, self.mask_padding_value)
        
        scores = (
            scores - scores.max(dim=-1, keepdim=True).values
        )  # Improve numerical stability
        att_weights = scores.softmax(dim=-1)
        if self.use_dropout:
            att_weights = self.dropout(att_weights)

        att = contract("bhij,bhjd->bhid", att_weights, v)
        att = att.transpose(1, 2)
        att = att.reshape(B, N, C)

        return att, att_weights

    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)  # Ensure directory exists

        torch.save(self.w.weight.data.cpu(), os.path.join(path, "w.pt"))

        if self.bias:
            torch.save(self.w.bias.data.cpu(), os.path.join(path, "w_bias.pt"))