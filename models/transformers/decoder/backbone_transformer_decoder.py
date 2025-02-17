import torch
import torch.nn as nn

import sys

sys.path.append(".")

from models.transformers.embeddings.backbone_embedding import BackboneEmbedding
from models.transformers.layers.backone_layer import BackboneTransformerLayer

class BackboneTransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        embedding_norm_scalar: float = 1.0,
        use_norm: bool = False,
        use_attention_dropout: bool = True,
        eps: float = 1e-6,
        mode: str = None,
        share_layers: bool = False,
        dtype: torch.dtype = torch.float64,
        embedding_type: str = "theoretical",
        device: str = "cpu",
        attention_type: str = "standard",
        num_embeddings: int = 1,
        max_length: int = 514
    ):

        super(BackboneTransformerDecoder, self).__init__()

        self.share_layers = share_layers
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.num_layers = num_layers
        self.device = device
        self.attention_type = attention_type

        # Initialize embedding layer
        #self.embedding = BackboneEmbedding(
        #    hidden_dim=hidden_dim,
        #    embedding_norm_scalar=embedding_norm_scalar,
        #    mode=mode,
        #    dtype=dtype,
        #    embedding_type=embedding_type,
        #    device=device,
        #    is_edge=(attention_type == "triangular"),
        #    num_embeddings=num_embeddings,
        #    max_length=max_length
        #)

        # Initialize transformer layers
        if share_layers:
            shared_layer = BackboneTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_norm=use_norm,
                use_attention_dropout=use_attention_dropout,
                eps=eps,
                dtype=dtype,
                device=device,
                attention_type=attention_type,
                is_edge=(attention_type == "triangular"),
            )
            self.layers = nn.ModuleList([shared_layer] * num_layers)
        else:
            # Use independent layers
            self.layers = nn.ModuleList([
                BackboneTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    use_norm=use_norm,
                    use_attention_dropout=use_attention_dropout,
                    eps=eps,
                    dtype=dtype,
                    device=device,
                    attention_type=attention_type,
                    is_edge=(attention_type == "triangular"),
                )
                for _ in range(num_layers)
            ])

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def forward(self, batch: dict) -> torch.Tensor:
        input_embd = self.embedding(batch=batch)
        hidden_state = input_embd
        
        all_attention_weights = []
        for layer in self.layers:
            if getattr(layer, 'gradient_checkpointing', False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_state, attention_weights = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_state, batch["batch_mask"]
                )
            else:
                hidden_state, attention_weights = layer(
                    hidden_state=hidden_state, batch_mask=None#batch["batch_mask"]
                )
            all_attention_weights.append(attention_weights)

        return hidden_state, all_attention_weights