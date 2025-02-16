
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
    def __init__(self, device: str = "cpu"):
        
        self.optimizer = optim.AdamW
        self.loss = PairwiseLoss
        self.lr = 1e-3
        self.epochs = 5
        self.batch_size = 2
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.collate_fn = backward_traversal_collate_fn

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
        loss_function = PairwiseLoss()

        learner = learner.train()

        for epoch in range(self.epochs):
            # Train model
            print(epoch)
            for batch in dataloader: # batch is a tuple of (example, signal) -> pairwise_batch is a tuple of (example1, example2, label) where label is signal1 > signal2
                batch_i,batch_j = batch
                # Forward pass
                output_si = learner(batch_i)
                output_sj = learner(batch_j)
                # Compute loss
                loss = loss_function(output_si[0],output_sj[0],batch_i["distance"])
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
        P_{i,j} =  1/(1+exp(-(M(s_i)-M(s_j))))
        \mathcal{L}_{pair} = -[y\log(P_{i,j})+(1-y)\log(1-P_{i,j})]
        y  = (1 if $r(x_i) > r(x_j)$ and 0 otherwise)
        """
        return - math.log(1/(1+math.exp(-(i-j))))*ij_distance # Entropy of i because  we assume it is the better state relative to j and that y = 1
    
    def trainWithTrajectory(self, trajectory, usage_quota=1):
        """
        Train the model using a successful trajectory that follows a near-optimal path.
        
        Parameters:
          trajectory: a list of state tensors representing the trajectory.
          usage_quota: float in (0, 1] determining the fraction of the trajectory to use.
        
        Returns:
          The average pairwise loss for the trajectory.
        """
        n = len(trajectory)
        if n < 2:
            return None  # Not enough states to compare
        
        # Determine the number of states to use based on usage_quota.
        num_states = max(2, int(n * usage_quota))
        # Select evenly spaced indices from the trajectory.
        indices = np.linspace(0, n - 1, num_states, dtype=int)
        selected_states = [trajectory[i] for i in indices]
        states_tensor = torch.stack(selected_states).to(self.device)
        
        # Forward pass: get ranking scores M(s)
        # Assume learner outputs shape (num_states, 1) so we squeeze to (num_states,)
        scores = self.learner(states_tensor).squeeze()
        
        total_loss = 0.0
        count = 0
        # For each pair (i, j) with i < j (i.e. j is closer to the goal), compute the loss.
        for i in range(len(scores) - 1):
            for j in range(i + 1, len(scores)):
                distance = j - i  # Use the difference in trajectory indices as a weight.
                loss_ij = self.pairwise_loss(scores[j], scores[i], distance)
                total_loss += loss_ij
                count += 1
        
        if count > 0:
            avg_loss = total_loss / count
            optimizer = self.optimizer()(self.learner.parameters(), lr=self.lr)
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
            return avg_loss.item()
            
        return None