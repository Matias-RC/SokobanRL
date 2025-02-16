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
                position_size: int = 200): # max length of input (e.g., grid + CLS)
        
        super(ScoringEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_norm_scalar = embedding_norm_scalar
        self.dtype = dtype
        self.device = device

        self.states_embd = nn.Embedding(
            num_embeddings=vocab_states_size + 2, # +2 for CLS and padding
            embedding_dim=hidden_dim,
            dtype=dtype,
            device=device
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=position_size, # max length of a flattened input
            embedding_dim=hidden_dim,
            dtype=dtype,
            device=device
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim,device=device)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass for embedding.

        Args:
            batch: Dictionary containing:
                - grid_si: Tensor of shape (batch_size, height, width) for state i.
                - grid_sj: Tensor of shape (batch_size, height, width) for state j.
        
        Returns:
            A tuple of embeddings for grid_si and grid_sj.
        """
        grid_si = batch["grid_si"]
        grid_sj = batch["grid_sj"]

        # Prepare inputs for grid_si and grid_sj
        inputs_si = self._prepare_inputs(grid_si)
        inputs_sj = self._prepare_inputs(grid_sj)

        # Compute embeddings for grid_si
        token_embeddings_si = self.states_embd(inputs_si["input_ids"])
        position_embeddings_si = self.position_embedding(inputs_si["position_ids"])
        embeddings_si = token_embeddings_si + position_embeddings_si

        # Compute embeddings for grid_sj
        token_embeddings_sj = self.states_embd(inputs_sj["input_ids"])
        position_embeddings_sj = self.position_embedding(inputs_sj["position_ids"])
        embeddings_sj = token_embeddings_sj + position_embeddings_sj

        # Normalize embeddings
        embeddings_si = self.layer_norm(embeddings_si) * self.embedding_norm_scalar
        embeddings_sj = self.layer_norm(embeddings_sj) * self.embedding_norm_scalar

        return embeddings_si, embeddings_sj

    def _prepare_inputs(self, grid: Tensor) -> Dict[str, Tensor]:
        """
        Prepare input_ids and position_ids dynamically from the grid.

        Args:
            grid: Tensor of shape (batch_size, height, width) representing the input grid.

        Returns:
            Dictionary containing:
                - input_ids: Encoded grid values, flattened and prefixed with CLS token.
                - position_ids: Position indices for each token in the flattened grid.
        """
        batch_size, height, width = grid.shape

        # Flatten the grid
        flattened_grid = grid.view(batch_size, -1)  # Shape: (batch_size, height * width)

        # Add CLS token (assume CLS token ID is 0)
        cls_token = torch.zeros((batch_size, 1), dtype=torch.long, device=grid.device)
        input_ids = torch.cat([cls_token, flattened_grid], dim=1)

        # Generate position IDs
        position_ids = torch.arange(input_ids.shape[1], device=grid.device).unsqueeze(0).repeat(batch_size, 1)

        return {"input_ids": input_ids, "position_ids": position_ids}

"""
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        
        input_ids = batch["input_ids"].to(self.device).flatten(1)
        #position_ids = batch["position_ids"]
        #types_ids = batch["types_ids"]
        
        token_embeddings = self.states_actions_embd(input_ids)
        #types_embeddings = self.type_sentence_embedding(types_ids)
        #position_embeddings = self.position_embedding(position_ids)
        
        embeddings = token_embeddings.to(self.dtype) #+ types_embeddings + position_embeddings 
        
        #embeddings = self.layer_norm(embeddings)
        #embeddings *= self.embedding_norm_scalar

        return embeddings
"""