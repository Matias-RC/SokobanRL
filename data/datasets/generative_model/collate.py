import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(batch):

    #encoder
    max_shape = max(item["shape_encoder"] for item in batch)
    padding = [max_shape - item["shape_encoder"] for item in batch]

    B, N = len(batch), max_shape
    
    grid_padded = torch.stack([
        F.pad(item["encoder_x"], (0, pad, 0, pad), value=0)
        for item, pad in zip(batch, padding)
    ])
    encoder_x = grid_padded.squeeze(-1)

    padding_masks = torch.stack([
        F.pad(
            torch.ones_like(item["encoder_x"], dtype=torch.bool), (0, pad, 0, pad), value=False
        )
        for item, pad in zip(batch, padding)
    ])

    attention_mask_encoder = ~padding_masks
    attention_mask_encoder = attention_mask_encoder.long()

    shapes_squared = torch.tensor([item["shape_encoder"] ** 2 for item in batch])

    position = torch.arange(max_shape**2).unsqueeze(0).expand(len(batch), -1)
    n = torch.tensor(max_shape**2).repeat(len(batch)) ** 0.5
    n_expanded = n.unsqueeze(1).expand(-1, max_shape**2)
    pos_i = position // n_expanded
    pos_j = position % n_expanded

    #decoder 
    max_shape = max(item["shape_decoder"] for item in batch)
    padding = [max_shape - item["shape_decoder"] for item in batch]

    decoder_x = torch.stack([
        F.pad(item["decoder_x"], (0, pad), mode="constant", value=0)
        for item, pad in zip(batch, padding)
    ])
    decoder_y = torch.stack([
        F.pad(item["decoder_y"], (0, pad), mode="constant", value=-100)
        for item, pad in zip(batch, padding)
    ])

    attention_mask_decoder = torch.stack([
        F.pad(
            torch.ones(item["shape_decoder"], dtype=torch.int64),
            (0, pad),
            mode="constant",
            value=0,)
        for item, pad in zip(batch, padding)
    ])

    position = torch.arange(max_shape).unsqueeze(0).expand(len(batch), -1)
    
    batch  = {
        "encoder_input_ids": encoder_x.unsqueeze(-1),
        "decoder_input_ids": decoder_x.unsqueeze(-1),
        "decoder_target_ids": decoder_y.unsqueeze(-1),
        
        "batch_mask_encoder": {
            "attention_mask": attention_mask_encoder == 1 ,
            "mask": None,
        },
        "batch_mask_decoder": {
            "attention_mask": attention_mask_decoder == 1 ,
            "mask": None,
        },
        "shape": shapes_squared,
        "pos_i": pos_i.reshape(B, N, N, 1),
        "pos_j": pos_j.reshape(B, N, N, 1),
    }

    return batch