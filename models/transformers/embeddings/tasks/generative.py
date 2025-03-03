from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GenerativeEmbedding(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 embedding_norm_scalar: float = 1.0, 
                 dtype: torch.dtype = torch.float32, 
                 device: torch.device = None,
                 vocab_states_size: int = 6, # wall, box, goal, player, box_on_goal, player_on_goal
                 vocab_actions_size: int = 4,
                 block_size: int = 64,
                 is_encoder = True): # max length of input (e.g., grid + CLS)
        
        super(GenerativeEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device
        self.is_encoder = is_encoder

        if is_encoder:
            self.in_proj = 3
            self.embedding = nn.Embedding( #states and actions
                num_embeddings=vocab_states_size + vocab_actions_size + 2, # +2 for CLS and padding
                embedding_dim=hidden_dim,
                dtype=dtype,
                device=device
            )
        else:
            self.in_proj = 1
            self.embedding = nn.Embedding( # only actions
                num_embeddings= vocab_actions_size + 2, 
                embedding_dim=hidden_dim,
                dtype=dtype,
                device=device
            )
        
        self.position_embedding = nn.Embedding( #states and actions
            num_embeddings=block_size, # +2 for CLS and padding
            embedding_dim=hidden_dim,
            dtype=dtype,
            device=device
        )
        
        self.projection = nn.Linear(in_features=self.in_proj,
                                    out_features=hidden_dim,
                                    dtype=dtype,
                                    device=device)
        

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        
        input_ids = batch["input_ids"].to(self.device).flatten(1)
        pos_i = batch["pos_i"].to(self.device).flatten(1)
        pos_j = batch["pos_j"].to(self.device).flatten(1)
        
        token_embeddings = self.embedding(input_ids)
        
        if self.is_encoder:
            embeddings = torch.concat([token_embeddings,
                                       pos_i.unsqueeze(-1),
                                       pos_j.unsqueeze(-1)
                                    ], dim=-1)
        else:
            positions = batch["positions"].to(self.device).flatten(1)
            token_embeddings = token_embeddings + self.position_embedding(positions)
            embeddings = torch.concat([token_embeddings, ],
                                      dim=-1)
       
        out_embedding = self.projection(embeddings)

        return out_embedding
