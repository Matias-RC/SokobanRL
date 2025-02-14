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
                vocab_actions_size: int = 4, # up,left,down,right
                vocab_states_size: int = 6, # wall, box, goal, player, box_on_goal, player_on_goal
                types_embedding_size: int = 2, # maps and actions
                position_size: int= 200): #max lenght of input
        
        super(ScoringEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_norm_scalar = embedding_norm_scalar
        self.dtype = dtype
        self.device = device

        
        self.states_actions_embd = nn.Embedding(num_embeddings=vocab_actions_size+vocab_states_size+3, #+3 for padding,CLS,SEP 
                                               embedding_dim=hidden_dim,
                                               dtype=dtype, device=device)
        self.position_embedding = nn.Embedding(num_embeddings=position_size, # max lenght of an input
                                               embedding_dim=hidden_dim,
                                               dtype=dtype, device=device)
        self.type_sentence_embedding = nn.Embedding(num_embeddings=types_embedding_size, # 0 for map inputs, 1 for actions inputs
                                               embedding_dim=hidden_dim,
                                               dtype=dtype, device=device)  
        
        self.layer_norm = nn.LayerNorm(hidden_dim)


    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        types_ids = batch["types_ids"]
        
        token_embeddings = self.states_actions_embd(input_ids)
        types_embeddings = self.type_sentence_embedding(types_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        embeddings = token_embeddings + types_embeddings + position_embeddings 
        
        embeddings = self.layer_norm(embeddings)
        embeddings *= self.embedding_norm_scalar

        return embeddings