import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(batch):

    max_shape = max(item["shape"] for item in batch)
    padding = [max_shape - item["shape"] for item in batch]

    B, N = len(batch), max_shape
    
    grid_padded = torch.stack([
        F.pad(item["grid"], (0, pad, 0, pad), value=0)
        for item, pad in zip(batch, padding)
    ])
    grid_padded = grid_padded.squeeze(-1)

    padding_masks = torch.stack([
        F.pad(
            torch.ones_like(item["grid"], dtype=torch.bool), (0, pad, 0, pad), value=False
        )
        for item, pad in zip(batch, padding)
    ])

    attention_mask = ~padding_masks
    attention_mask = attention_mask.long()

    shapes_squared = torch.tensor([item["shape"] ** 2 for item in batch])

    rank = torch.stack([ item["rank"]  for item in batch ] )

    position = torch.arange(max_shape**2).unsqueeze(0).expand(len(batch), -1)
    n = torch.tensor(max_shape**2).repeat(len(batch)) ** 0.5
    n_expanded = n.unsqueeze(1).expand(-1, max_shape**2)
    pos_i = position // n_expanded
    pos_j = position % n_expanded

    o  = {
        "input_ids": grid_padded.unsqueeze(-1),
        "batch_mask": {
            "attention_mask": attention_mask == 1 ,
            "mask": None,
        },
        "rank": rank,
        "shape": shapes_squared,
        "pos_i": pos_i.reshape(B, N, N, 1),
        "pos_j": pos_j.reshape(B, N, N, 1),
    }

    return o