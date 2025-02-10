
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import random

# DeepPairwiseRankAggregation
class DPRA:
    def __init__(self, device: str = "cpu"):
        
        self.optimizer = optim.AdamW
        self.loss = nn.BCELoss
        self.lr = 1e-3
        self.epochs = 100
        self.batch_size = 32
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def do(self, dataset, model):

        learner = model.get_learner()
        learner.is_training = True
        
        self.fit(dataset, learner)

        learner.is_training = False

        return model.set_learner(learner)
    
    def fit(self, dataset, learner):

        optimizer = self.optimizer()(learner.parameters(), lr=self.lr)
        loss_function = self.loss()

        for epoch in range(self.epochs):
            # Train model
            for batch in dataset: # batch is a tuple of (example, signal) -> pairwise_batch is a tuple of (example1, example2, label) where label is signal1 > signal2
                # Forward pass
                output = learner(batch)
                # Compute loss
                loss = loss_function(output, batch)
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()
                # Zero gradients
                optimizer.zero_grad()