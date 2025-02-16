import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import numpy as np

def collate_fn(batch):
    """
    Expects each element in batch to be a dict with keys:
      - "grid_si": Tensor (1D, representing tokenized grid with CLS token already added or to be added)
      - "grid_sj": Tensor (1D, same as above)
      - "distance": scalar (ranking label)
    Pads sequences to the max length and creates an attention mask.
    Returns a dict with keys:
      - "si": Tensor of shape (batch_size, max_length)
      - "sj": Tensor of shape (batch_size, max_length)
      - "attention_mask": Tensor of shape (batch_size, max_length) (1 for valid tokens, 0 for padded)
      - "distance": Tensor of shape (batch_size,)
    """
    si_list = []
    sj_list = []
    distance_list = []
    
    for item in batch:
        # Allow for either key naming ("grid_si"/"grid_sj" or "si"/"sj")
        si = item.get("grid_si", item.get("si"))
        sj = item.get("grid_sj", item.get("sj"))
        # If the grid is 2D, flatten it.
        if si.dim() > 1:
            si = si.view(-1)
        if sj.dim() > 1:
            sj = sj.view(-1)
        si_list.append(si)
        sj_list.append(sj)
        distance_list.append(item["distance"])
    
    # Determine maximum length
    max_length = max(si.size(0) for si in si_list)
    
    padded_si = []
    padded_sj = []
    attention_mask = []
    
    for si, sj in zip(si_list, sj_list):
        pad_len = max_length - si.size(0)
        # Pad with zeros (assumes that 0 is the pad token, adjust if necessary)
        padded_si.append(torch.cat([si, torch.zeros(pad_len, dtype=si.dtype)]))
        padded_sj.append(torch.cat([sj, torch.zeros(pad_len, dtype=sj.dtype)]))
        mask = torch.cat([torch.ones(si.size(0), dtype=torch.long),
                          torch.zeros(pad_len, dtype=torch.long)])
        attention_mask.append(mask)
    
    batch_si = torch.stack(padded_si, dim=0)
    batch_sj = torch.stack(padded_sj, dim=0)
    batch_attention_mask = torch.stack(attention_mask, dim=0)
    batch_distance = torch.tensor(distance_list, dtype=torch.float)
    
    return {"si": batch_si, "sj": batch_sj, "attention_mask": batch_attention_mask, "distance": batch_distance}

