import math
import random 
import numpy as np
import torch


class Replayer:

    def __init__(self, agent, method):

        self.agent = agent
        if method == "pairwise_loss":
            self.train_with_trayectory = self.trainWithTrajectory
        else:
            self.train_with_trayectory = self.trainWithTrajectory

    def do(self, succesful_trajectories):
        for t in succesful_trajectories:
            loss = self.train_with_trayectory(t)
            
    @staticmethod
    def trainWithTrajectory(model, trajectory, num_comparisons=10):
        """
        Train the model using a successful trajectory with a fixed number of comparisons.

        Parameters:
          model: The PyTorch model to be trained.
          trajectory: A list of state tensors representing the trajectory.
          num_comparisons: The fixed number of state pairs to use for training.

        Returns:
          The average pairwise loss for the trajectory.
        """
        n = len(trajectory)
        if n < 2:
            return None  # Not enough states to compare

        # Ensure at least `num_comparisons` pairs can be formed.
        num_states = min(n, num_comparisons + 1)
        indices = np.linspace(0, n - 1, num_states, dtype=int)
        selected_states = [trajectory[i] for i in indices]
        states_tensor = torch.stack(selected_states).to(model.device)

        # Forward pass: get ranking scores M(s)
        scores = model.learner(states_tensor).squeeze()

        total_loss = 0.0
        count = 0
        optimizer = model.optimizer()(model.learner.parameters(), lr=model.lr)
        optimizer.zero_grad()

        # For a fixed number of comparisons, randomly sample pairs (i, j) with i < j.
        comparisons = []
        for _ in range(num_comparisons):
            i, j = np.random.choice(len(scores), size=2, replace=False)
            if i > j:
                i, j = j, i  # Ensure i < j
            comparisons.append((i, j))

        for i, j in comparisons:
            distance = j - i  # Use the difference in trajectory indices as a weight.
            loss_ij = model.pairwise_loss(scores[j], scores[i], distance)
            total_loss += loss_ij
            count += 1

        if count > 0:
            avg_loss = total_loss / count
            avg_loss.backward()
            optimizer.step()
            return avg_loss.item()

        return None

    def do(self, successful_trajectories, model, num_comparisons=10):
        """
        Perform training on a set of successful trajectories.

        Parameters:
          successful_trajectories: A list of trajectories (each a list of state tensors).
          model: The PyTorch model to be trained.
          num_comparisons: The fixed number of comparisons to use for each trajectory.
        """
        for t in successful_trajectories:
            loss = self.train_with_trajectory(model, t, num_comparisons=num_comparisons)
            print(f"Loss for trajectory: {loss}")
