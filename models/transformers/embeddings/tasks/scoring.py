from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ScoringEmbedding(nn.Module):
    def __init__(self, 
                hidden_dim:int, 
                embedding_norm_scalar: float = 1.0, 
                dtype: torch.dtype = torch.float32, 
                device: torch.device = None, 
                position_size: int= 514):
        
        super(ScoringEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_norm_scalar = embedding_norm_scalar
        self.dtype = dtype
        self.device = device

        # Initialize embeddings
        #self.token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_dim, dtype=dtype, device=device)  # 30522 is typical for BERT-like vocab
        self.position_embedding = nn.Embedding(num_embeddings=position_size, embedding_dim=hidden_dim, dtype=dtype, device=device)  # Supports up to 514 tokens
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Move embeddings to the specified device
        self.to(dtype=self.dtype, device=self.device)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        
        input_ids = batch["input_ids"]
        device = input_ids.device
        
        #self.token_embedding = self.token_embedding.to(device)
        self.position_embedding = self.position_embedding.to(device)
        #token_embeddings = self.token_embedding(input_ids)
        
        position_ids = batch["position_ids"]
        position_embeddings = self.position_embedding(position_ids)
        
        embeddings = position_embeddings #token_embeddings + position_embeddings
        
        embeddings = self.layer_norm(embeddings)
        embeddings *= self.embedding_norm_scalar

        return embeddings