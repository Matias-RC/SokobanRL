import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, List
import random
import math
from managers.sokoban_manager import Node, SokobanManager
from managers.inverse_manager import InvertedNode, InversedSokobanManager
from simAnnaeling import *
from SokoSource import PriorityQueue

def create_environment(board_shape, posWalls, posGoals, key):

    board = np.zeros(board_shape, dtype=int)

    for r, c in posWalls:
        board[r, c] = 1
    for r, c in posGoals:
        board[r, c] = 4
    player_pos, box_positions = key
    for r, c in box_positions:
        if board[r, c] == 4:
            board[r, c] = 5
        else:
            board[r, c] = 3
    pr, pc = player_pos
    if board[pr, pc] == 4:
        board[pr, pc] = 6
    else:
        board[pr, pc] = 2
        
    return board
# ================================
# Mock Delta Scorer
# ================================
class DummyDeltaScorer():
    def __init__(self, posGoals, posWalls, gridDims):
        self.posGoals = posGoals
        self.posWalls = posWalls
        self.gridDims = gridDims
    
    # I want to set an input so that I can direcly control the output (simulating the delta)
    def printGrid(self, state):
        #use the function create_environment to create the grid
        board = create_environment(self.gridDims, self.posWalls, self.posGoals, state)
        print(board)
        return board
        
    def m(self, state, action_sequence):
        print("m")
        self.printGrid(state)
        print(action_sequence)
        return float(input("Enter the delta: "))
    
    def q(self, node, legal_actions):
        print("q")
        # I want to set an input so that I can direcly control the output (simulating the delta)
        self.printGrid(node.state)
        print(legal_actions)
        probs = []
        for i in range(len(legal_actions)):
            print(f"{i}: {legal_actions[i]}")
            prob = input("Enter the probability: ")
            probs.append(float(prob))
        return probs

# ================================
# Test Setup
# ================================

grid = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 3, 0, 4, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 2, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1]
])



library =[
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

instanceWrapper = SokobanManager()
node = instanceWrapper.initializer(grid)
initial_solution = [[(-1, 0)], [(-1, 0)], [(-1, 0)], [(0, 1)], [(0, 1)]]
posPlayer, posBox = instanceWrapper.PosOfPlayer(grid), instanceWrapper.PosOfBoxes(grid)
for action in initial_solution:
    condition, new_node = instanceWrapper.LegalUpdate(action, node.state, node)
    if condition:
        node = new_node
    else:
        break

#trajectories_cache, annealing_data =simulated_annealing_trajectory(initial_solution=node, grid=grid,
#                               manager=SokobanManager(), move_library=library, 
#                               alternative_generator_fn=alternative_generator, 
#                               delta_scorer=DummyDeltaScorer(instanceWrapper.posGoals, instanceWrapper.posWalls, grid.shape),
#                               perceived_improvability_fn=perceived_improvability,
#                               num_alternatives=1)
#
#for t in trajectories_cache:
#    print(t.trajectory())
inverseManager = InversedSokobanManager()
endNode = InvertedNode(((1,3),((1,4))))

model  =  DummyDeltaScorer(instanceWrapper.posGoals,instanceWrapper.posWalls,grid.shape)

def backward_traversal_worst_paths(inverseManager, end_node, initial_grid, max_depth, max_frontier_capacity=1000000):
    final_grid_state = inverseManager.initializer(initial_grid=initial_grid, end_node=end_node)
    # Create an InvertedNode from end_node.
    end_node = InvertedNode(state=end_node.state, parent=None, action=None, inversed_action=None, rank=0)
    frontier = PriorityQueue()
    frontier.push(end_node, 0)
    depth = max_depth
    seen_states = set()
    while not frontier.isEmpty() and frontier.Count < max_frontier_capacity and depth > 0:
        node = frontier.pop()
        seen_states.add(node.state)
        for m in [[(-1,0)],[(1,0)],[(0,-1)],[(0,1)]]:
            conditition,  new_node = inverseManager.legalInvertedUpdate(m,node.state,node)
            if conditition and new_node.state not in seen_states:
                node.children.append(new_node)
                frontier.push(new_node,(-model.m(new_node.state)+1)*20+len(new_node.trayectory()))
        depth -= 1
    print("Count:", frontier.Count)
    return frontier, end_node

frontier, end_node = backward_traversal_worst_paths(inverseManager,endNode, grid, 5,100000)

print(frontier)