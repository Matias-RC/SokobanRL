import torch
import torch.nn as nn

from models.transformers.embeddings.tasks.scoring import ScoringEmbedding


class TheoreticalEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_norm_scalar: float = 1.0,
        mode: str = None,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        is_edge: bool = False,
    ):
        super(TheoreticalEmbedding, self).__init__()

        self.mode = mode

        if self.mode == "scoring":
            self.embedding = ScoringEmbedding(
                hidden_dim=hidden_dim,
                embedding_norm_scalar=embedding_norm_scalar,
                dtype=dtype,
                device=device
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def forward(self, batch: dict) -> torch.Tensor:

        return self.embedding(batch)