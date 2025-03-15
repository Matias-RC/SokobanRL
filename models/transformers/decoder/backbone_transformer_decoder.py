import torch
import torch.nn as nn

import sys

sys.path.append(".")

from models.transformers.embeddings.backbone_embedding import BackboneEmbedding
from models.transformers.layers.backbone_encoder_decoder_layer import BackboneTransformerLayer

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
        block_size: int = 128,
        masked_multihead_attention: bool = False,
        is_cross_attention: bool = False,
    ):

        super(BackboneTransformerDecoder, self).__init__()

        self.share_layers = share_layers
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.num_layers = num_layers
        self.device = device
        self.attention_type = attention_type

        self.embedding = BackboneEmbedding( hidden_dim=hidden_dim,
                                            embedding_norm_scalar=embedding_norm_scalar,
                                            mode=mode,
                                            dtype= torch.float64,
                                            embedding_type=embedding_type,
                                            device=device,
                                            block_size=block_size,
                                            is_encoder = False,
                                           )

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
                block_size=block_size,
                masked_multihead_attention=masked_multihead_attention,
                is_cross_attention=is_cross_attention,
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
                    block_size=block_size,
                    masked_multihead_attention=masked_multihead_attention,
                    is_cross_attention=is_cross_attention,
                )
                for _ in range(num_layers)
            ])

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def forward(self, batch: dict = None) -> torch.Tensor:
        input_embd = self.embedding(batch=batch) #decoder inputs that pass throug embedding
        hidden_state = input_embd
        cross_hidden_states = batch["cross_hidden_states"]
        
        all_attention_weights = []
        for layer in self.layers:
            if getattr(layer, 'gradient_checkpointing', False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_state, attention_weights = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_state, batch["batch_mask_decoder"]
                )
            else:
                hidden_state, attention_weights = layer(
                    query_hidden_states=hidden_state, cross_hidden_states=cross_hidden_states, batch_mask=batch["batch_mask_decoder"]
                )
            all_attention_weights.append(attention_weights)

        return hidden_state, all_attention_weights