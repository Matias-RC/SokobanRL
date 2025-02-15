import torch
from torch.utils.data import Dataset
import numpy as np
import random

class BackwardTraversalDataset(Dataset):
    def __init__(self,dataset,one_batch=True):
        """
        Args:
            dataset (list): List of batches
            batch (list): List of tuples of the form (grid, rank)
            one_batch (bool): If True, the dataset is a single batch
        Output:
            A dataset of the form (grid, rank) as torch tensors
        """
        if one_batch:
            batch = dataset[0]
            for example in batch:
                grid, rank = example
                grid = torch.tensor(grid, dtype=torch.float32)
                rank = torch.tensor(rank, dtype=torch.float32)
                self.dataset.append((grid, rank))
        else:
            for batch in dataset:
                for example in batch:
                    grid, rank = example
                    grid = torch.tensor(grid, dtype=torch.float32)
                    rank = torch.tensor(rank, dtype=torch.float32)
                    self.dataset.append((grid, rank))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        grid, rank = self.dataset[idx]
        return {"grid": grid, "rank": rank}