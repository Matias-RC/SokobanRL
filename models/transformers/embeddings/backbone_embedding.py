import torch
import torch.nn as nn

from models.transformers.embeddings.learnable import LearnableEmbedding

class BackboneEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_norm_scalar: float = 1.0,
        mode: str = None,
        embedding_type: str = "theoretical",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        is_edge: bool = False,
        num_embeddings: int = 1,
        max_length: int = 514,
    ):
        super(BackboneEmbedding, self).__init__()

        if embedding_type == "learnable":
            self.embedding = LearnableEmbedding(
                hidden_dim=hidden_dim,
                embedding_norm_scalar=embedding_norm_scalar,
                mode=mode,
                dtype=dtype,
                device=device,
                num_embeddings=num_embeddings,
                max_length=max_length
            )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.embedding(batch)