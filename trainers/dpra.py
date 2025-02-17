
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from collections import defaultdict
import random
from SokoSource import final_state_grid
from data_generators.collate_fn import collate_fn
from torch.utils.data import Dataset, DataLoader
from src.loss_function.pairwise_loss import PairwiseLoss
from data.datasets.backward_traversal.collate import collate_fn as backward_traversal_collate_fn




# DeepPairwiseRankAggregation
class DPRA:
    def __init__(self, device: str = "cpu", verbose: bool = True):
        
        self.optimizer = optim.AdamW
        self.loss = PairwiseLoss()
        self.lr = 1e-3
        self.epochs = 1000
        self.batch_size = 2
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.collate_fn = backward_traversal_collate_fn
        self.verbose = verbose

    def do(self, dataset, model):

        learner = model.get_learner()
        learner.is_training = True
        
        for task in dataset: # Change it should only be for the current session 
            dataloader = DataLoader(task,
                                    shuffle = False,
                                    batch_size=self.batch_size,
                                    collate_fn=self.collate_fn)
                
            self.fit(dataloader, learner)

        learner.is_training = False

        return model.set_learner(learner)
    
    def fit(self, dataloader, learner):
        optimizer = self.optimizer(learner.parameters(), lr=self.lr)
        learner = learner.train()

        for epoch in range(self.epochs):
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}")

            total_loss = 0
            batch_count = 0

            for i, batch in enumerate(dataloader):  
                output, _ = learner(batch)
                loss = self.loss(output, batch["rank"])
                total_loss += loss.item()
                batch_count += 1

                # if self.verbose:
                #     print(f"  Batch {i+1}: Loss = {loss.item():.4f}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            if self.verbose:
                print(f"  -> Average Loss for Epoch {epoch+1}: {avg_loss:.4f}\n")
        
