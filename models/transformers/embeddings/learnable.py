import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformers.embeddings.tasks.scoring import ScoringEmbedding

class LearnableEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_norm_scalar: float = 1.0,
        mode: str = None,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        num_embeddings: int = 10,
        max_length: int = 514,
    ):
        super(LearnableEmbedding, self).__init__()
    
        self.mode = mode

        if self.mode == "scoring":
                self.embedding = ScoringEmbedding(
                    hidden_dim=hidden_dim,
                    embedding_norm_scalar=embedding_norm_scalar,
                    dtype=dtype,
                    device=device,
                    vocab_actions_size=num_embeddings,
                    vocab_states_size=num_embeddings,
                    position_size=max_length
                )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.embedding(batch)