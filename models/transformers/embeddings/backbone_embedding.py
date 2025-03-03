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
        block_size: int = 514,
        is_encoder = True,
    ):
        super(BackboneEmbedding, self).__init__()

        if embedding_type == "learnable":
            self.embedding = LearnableEmbedding(
                hidden_dim=hidden_dim,
                embedding_norm_scalar=embedding_norm_scalar,
                mode=mode,
                dtype=dtype,
                device=device,
                block_size=block_size,
                is_encoder = is_encoder,
            )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.embedding(batch)