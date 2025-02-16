import torch
import torch.nn as nn

from models.transformers.encoder.backbone_transformer_encoder import BackboneTransformerEncoder
from models.transformers.feed_forward_networks.seqregressor import SequenceRegressor

class TransformerEncoderForScoring(nn.Module):
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
        share_layers: bool = False,
        device: str = "cpu",
        embedding_type: str = "theoretical",
        attention_type: str = "standard",
        output_dim: int = 1,
    ) -> None:

        super(TransformerEncoderForScoring, self).__init__()

        self.name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.__class__.__name__.lower(),
            hidden_dim,
            num_layers,
            num_heads,
            dropout_rate,
            embedding_norm_scalar,
            use_norm,
            share_layers,
            embedding_type,
            attention_type,
        )

        #process all the input data with a decoder transformer
        self.encoder = BackboneTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            embedding_norm_scalar=embedding_norm_scalar,
            use_norm=use_norm,
            use_attention_dropout=use_attention_dropout,
            eps=eps,
            share_layers=share_layers,
            device=device,
            mode="scoring",
            embedding_type=embedding_type,
            attention_type=attention_type,
        )

        #regressor
        self.regressor = SequenceRegressor(hidden_dim=hidden_dim,
                                           output_dim=1,
                                           dropout_rate=dropout_rate,
                                           device=device)

        self.is_training = False

    def forward(self, x) -> torch.Tensor:

        if self.is_training:
            return self.structured_forward(x)
        else:
            return self.unstructured_forward(x)

    def unstructured_forward(self, instance) -> torch.Tensor:

        x = self.to_tensor(instance) # similar to the collate function in the dataloader

        return self.forward(x)

    def structured_forward(self, x: torch.Tensor) -> torch.Tensor:

        activations, attn_weights = self.encoder(x)

        y_hat = self.regressor(activations) # regressor is a feed forward network

        return y_hat, activations
    
    def to_tensor(self, instance) -> torch.Tensor:
        # Convert instance to tensor similar to the collate function in the dataloader
        return None