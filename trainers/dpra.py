
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from collections import defaultdict
import random
#from data_generators.collate_fn import collate_fn
from torch.utils.data import Dataset, DataLoader

# DeepPairwiseRankAggregation
class DPRA:
    def __init__(self, device: str = "cpu"):
        
        self.optimizer = optim.AdamW
        self.loss = nn.BCELoss
        self.lr = 1e-3
        self.epochs = 100
        self.batch_size = 32
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        #self.collate_fn = collate_fn

    def do(self, dataset, model):

        learner = model.get_learner()
        learner.is_training = True

        #dataloader = DataLoader(dataset,
        #                        shuffle = True,
        #                        collate_fn=self.collate_fn)
        
        self.fit(dataset, learner)

        learner.is_training = False

        return model.set_learner(learner)
    
    def fit(self, dataloader, learner):

        optimizer = self.optimizer()(learner.parameters(), lr=self.lr)
        loss_function = self.loss()

        for epoch in range(self.epochs):
            # Train model
            for batch in dataloader: # batch is a tuple of (example, signal) -> pairwise_batch is a tuple of (example1, example2, label) where label is signal1 > signal2
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
    
    def pairwise_loss(self, i,j, ij_distance):
        """
        Compute the pairwise loss of the model with the succesful trajectories
        Theory:
        \mathcal{L}_{pair} = -[y\log(P_{i,j})+(1-y)\log(1-P_{i,j})]
        y  = (1 if $r(x_i) > r(x_j)$ and 0 otherwise)
        """
        return - math.log(1/(1+math.exp(-(i-j))))*ij_distance # Entropy of i because  we assume it is the better state relative to j and that y = 1

    def trainWithTrajectory():
        pass