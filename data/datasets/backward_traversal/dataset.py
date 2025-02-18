import torch
from torch.utils.data import Dataset
import numpy as np
import random

class BackwardTraversalDataset(Dataset):
    def __init__(self, batch):
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