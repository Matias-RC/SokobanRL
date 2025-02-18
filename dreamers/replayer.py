import math
import random 
import numpy as np
import torch


class Replayer:

    def __init__(self, agent, method):

        self.agent = agent
        if method == "pairwise_loss":
            self.train_with_trajectory = self.trainPairwise
        else:
            self.train_with_trajectory = self.trainPairwise
        self.posGoals = None

    def do(self, successful_trajectories, model, num_comparisons=10):
        for t in successful_trajectories:
            loss = self.train_with_trajectory(model, t, num_comparisons=num_comparisons)
            print(f"Loss for trajectory: {loss}")

        return model

    def PosOfGoals(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 4) | (grid == 5) | (grid == 6)))
    
    def matrix_state_grid(self, initial_grid, final_player_pos, final_pos_boxes):
        '''Creates the final grid from the initial grid and the final player and box positions.'''
        final_grid = np.copy(initial_grid) #copy
        final_grid[(final_grid == 2) | (final_grid == 3) | (final_grid == 5) | (final_grid == 6)] = 0 #reset
        
        if final_player_pos in self.posGoals:
            final_grid[final_player_pos] = 6  # Player on Button
        else:
            final_grid[final_player_pos] = 2  # Normal Player

        for box in list(final_pos_boxes):
            if box in self.posGoals:
                final_grid[box] = 5  # Box on Button
            else:
                final_grid[box] = 3  # Normal Box
        
        return final_grid

    def trainPairwise(self, model, task, num_comparisons=10):
        self.posGoals = self.PosOfGoals(task.initial_state)
        trajectory =  task.solution.statesList()

        n = len(trajectory)
        if n < 2:
            return None 

        # Ensure at least `num_comparisons` pairs can be formed.
        num_states = min(n, num_comparisons + 1)
        indices = np.linspace(0, n - 1, num_states, dtype=int)
        selected_states = [self.matrix_state_grid(task.initial_state ,trajectory[i][0],trajectory[i][1]) for i in indices]
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

