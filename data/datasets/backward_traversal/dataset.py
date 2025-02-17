import torch
from torch.utils.data import Dataset
import numpy as np
import random

class BackwardTraversalDataset(Dataset):
    def __init__(self, batch):
        """
        Args:
            dataset (list): List of batches
            batch (list): List of tuples of the form (grid, rank)
            one_batch (bool): If True, the dataset is a single batch
        Output:
            A dataset of the form (grid_i, grid_j, rank) as torch tensors
        """
        super().__init__()  # Ensures compatibility with torch Dataset
        
        self.dataset = self.contruct_examples(batch)
    
    def contruct_examples(self, batch):
        n = len(batch)
        indices = list(range(n))
        random.shuffle(indices)   

        examples = [batch[i] for i in indices]

        return examples
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "grid": torch.tensor(self.dataset[idx]["grid"], dtype=torch.long),  
            "rank": torch.tensor(self.dataset[idx]["rank"], dtype=torch.long),  
            "shape": self.dataset[idx]["grid"].shape[0]
        }