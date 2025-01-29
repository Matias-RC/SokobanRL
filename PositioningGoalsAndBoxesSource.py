import math
import random
import numpy as np
import torch
import torch.nn as nn


# Actions mapped to integers
ACTION_MAP = {
    0: (-1, 0),  # 'w' (UP)
    1: (1, 0),   # 's' (DOWN)
    2: (0, -1),  # 'a' (LEFT)
    3: (0, 1)    # 'd' (RIGHT)
}

# Arr : Arrengement of goals and boxes
# Batch : Set of arrengements

def NumberOfBatches(grid, c):
    y, x = grid
    return int(math.sqrt(y*x) + c)

def RandomArr(grid, num_goals):
    """Generates a random goal-box arrangement"""
    EmptySpace = np.where(grid == 0)
    for idx in range(num_goals):
        goal_position = random.choice(list(zip(EmptySpace[0], EmptySpace[1])))
        grid[goal_position] = 5
        EmptySpace = np.where(grid == 0)
    return grid

#Posible Finishing Positions
def PFP(grid):
    """Place player to the side of goals"""
    positions = []
    goals = np.where(grid == 5)
    goals = list(zip(goals[0], goals[1]))
    for i in goals:
        for j in range(4):
            dy, dx = ACTION_MAP[j]
            if grid[i[0]+dy, i[1]+dx] == 0:
                positions.append([i[0]+dy, i[1]+dx])
    return positions


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_hidden_layers):
        super(MLP, self).__init__()
        
        # List to hold layers
        layers = []
        
        # First hidden layer (input to first hidden layer)
        layers.append(nn.Linear(in_dim, hid_dim))
        layers.append(nn.ReLU())
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hid_dim, out_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
"""
Inversed actor critic:
    - with inverse logic we generate backwards steps for simplicity imagine that during training it is breadth first
    - then we train the ActorCritic to determinate what actions lead to bigger reward
    - We determinate the reward to be the bigger if for longer shortest paths.
    - Since we only want some basic pattern recognition the AC will only take sorrounding grid area from the player.
        -> Therfore the size of the input is constant  (padding if needed)
    
        A(s, a) = r(s,a,s') + V(s') - V(s)
        r(s, a, s') = Delta Box Lines
        V(s) = Length of Shortest path to succesful terminal state
"""

def ac(actor, critic, episodes, max_steps=10, lr_a=1e-3, lr_c=1e-3):
    return

