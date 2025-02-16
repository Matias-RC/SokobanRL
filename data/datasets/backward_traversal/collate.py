import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(batch):

    max_shape = max(item["shape"] for item in batch)
    padding = [max_shape - item["shape"] for item in batch]

    B, N = len(batch), max_shape

    grid_si_padded = torch.stack([
        F.pad(item["grid_si"], (0, pad, 0, pad), value=0)
        for item, pad in zip(batch, padding)
    ])
    grid_si_padded = grid_si_padded.squeeze(-1)

    grid_sj_padded = torch.stack([
        F.pad(item["grid_sj"], (0, pad, 0, pad), value=0)
        for item, pad in zip(batch, padding)
    ])
    grid_sj_padded = grid_sj_padded.squeeze(-1)

    padding_masks = torch.stack([
        F.pad(
            torch.ones_like(item["grid_si"], dtype=torch.bool), (0, pad, 0, pad), value=False
        )
        for item, pad in zip(batch, padding)
    ])
    attention_mask = ~padding_masks
    attention_mask = attention_mask.long()

    shapes_squared = torch.tensor([item["shape"] ** 2 for item in batch])

    distance = torch.stack([ item["distance"]  for item in batch ] )

    position = torch.arange(max_shape**2).unsqueeze(0).expand(len(batch), -1)
    n = torch.tensor(max_shape**2).repeat(len(batch)) ** 0.5
    n_expanded = n.unsqueeze(1).expand(-1, max_shape**2)
    pos_i = position // n_expanded
    pos_j = position % n_expanded

        
    o_i  = {
        "input_ids": grid_si_padded.unsqueeze(-1),
        "batch_mask": {
            "attention_mask": attention_mask == 1 ,  # attention_mask,
            "mask": None, #{"AB": window_mask_AB,
                    #"BC": window_mask_BC,
                    #"CA": window_mask_CA,
                    #"QK": window_mask_QK},
        },
        "distance": distance,
        "shape": shapes_squared,
        "pos_i": pos_i.reshape(B, N, N, 1),
        "pos_j": pos_j.reshape(B, N, N, 1),
    }

    o_j  = {
        "input_ids": grid_sj_padded.unsqueeze(-1),
        "batch_mask": {
            "attention_mask": attention_mask == 1 ,  # attention_mask,
            "mask": None, #{"AB": window_mask_AB,
                    #"BC": window_mask_BC,
                    #"CA": window_mask_CA,
                    #"QK": window_mask_QK},
        },
        "distance": distance,
        "shape": shapes_squared,
        "pos_i": pos_i.reshape(B, N, N, 1),
        "pos_j": pos_j.reshape(B, N, N, 1),
    }
    return o_i,o_j