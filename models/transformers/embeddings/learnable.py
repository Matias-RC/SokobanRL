import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformers.embeddings.tasks.scoring import ScoringEmbedding
from models.transformers.embeddings.tasks.generative import GenerativeEmbedding

class LearnableEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_norm_scalar: float = 1.0,
        mode: str = None,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        block_size: int = 514,
        is_encoder:bool = True,
    ):
        super(LearnableEmbedding, self).__init__()
    
        self.mode = mode

        if self.mode == "scoring":
                self.embedding = ScoringEmbedding(
                                                  hidden_dim=hidden_dim,
                                                  embedding_norm_scalar=embedding_norm_scalar,
                                                  dtype=dtype,
                                                  device=device,
                                                  block_size=block_size,
                )
        elif self.mode == "generative":
             self.embedding = GenerativeEmbedding(hidden_dim=hidden_dim,
                                                  embedding_norm_scalar=embedding_norm_scalar,
                                                  dtype=dtype,
                                                  device=device,
                                                  block_size=block_size,
                                                  is_encoder=is_encoder,
                                                 )
    
    def forward(self, batch: dict) -> torch.Tensor:
        return self.embedding(batch)
    