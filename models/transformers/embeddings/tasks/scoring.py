from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScoringEmbedding(nn.Module):
    def __init__(self, 
                hidden_dim: int, 
                embedding_norm_scalar: float = 1.0, 
                dtype: torch.dtype = torch.float32, 
                device: torch.device = None,
                vocab_states_size: int = 6, # wall, box, goal, player, box_on_goal, player_on_goal
                vocab_actions_size: int = 4,
                position_size: int = 200): # max length of input (e.g., grid + CLS)
        
        super(ScoringEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device

        self.states_actions_embd = nn.Embedding(
            num_embeddings=vocab_states_size + vocab_actions_size + 2, # +2 for CLS and padding
            embedding_dim=hidden_dim,
            dtype=dtype,
            device=device
        )

        self.projection = nn.Linear(
            in_features=hidden_dim+2,
            out_features=hidden_dim,
            dtype=dtype,
            device=device)
        

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        
        input_ids = batch["input_ids"].to(self.device).flatten(1)
        pos_i = batch["pos_i"].to(self.device).flatten(1)
        pos_j = batch["pos_j"].to(self.device).flatten(1)
        
        token_embeddings = self.states_actions_embd(input_ids)
        
        embeddings = torch.concat([
            token_embeddings,
            pos_i.unsqueeze(-1),
            pos_j.unsqueeze(-1)
        ],
            dim=-1)

        out_embedding = self.projection(embeddings)

        return out_embedding
