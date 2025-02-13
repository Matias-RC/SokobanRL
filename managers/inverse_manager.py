from turtle import pos
import numpy as np
from collections import defaultdict, deque
import random
from dataclasses import dataclass
from typing import Any, Optional, List



@dataclass
class InvertedNode:
    ''' Data structure for a node in the search tree. '''
    state: Any
    parent: Optional['InvertedNode'] = None
    children: Optional[List['InvertedNode']] = None
    action: Optional[Any] = None
    inversed_action: Optional[Any] = None
    rank: int = 0

    def trajectory(self) -> List['InvertedNode']:
        node, path = self, []

        while node:
            path.append(node.action)
            node = node.parent
            
        return list(path)#[1:]
    
    def statesList(self) -> List['InvertedNode']:
        node, path = self, []

        while node:
            path.append(node.state)
            node = node.parent
        return list(path)
    
    def nodesList(self) -> List['InvertedNode']:
        node, path = self, []
        while node:
            path.append(node)
            node = node.parent
        return list(path)
    
    def inversed_trajectory(self) -> List['InvertedNode']:
        node, path = self, []

        while node:
            path.append(node.inversed_action)
            node = node.parent
            
        return list(path)


class InversedSokobanManager:
    ''' Manager for sokoban that generate an inversed path.'''
    def __init__(self):
        self.posWalls = None
        self.posGoals = None
    
    def PosOfWalls(self, grid):
        return tuple(tuple(x) for x in np.argwhere(grid == 1))

    def PosOfGoals(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 4) | (grid == 5) | (grid == 6)))
    
    def PosOfBoxes(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 3) | (grid == 5)))
    
    def isEndState(self,node):
        '''Check if the position is winner state.'''
        return sorted(node.state[1]) == sorted(self.posGoals)
        
    def final_state_grid(self, initial_grid, final_player_pos, final_pos_boxes):
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
    """
    def isFailed(self, posPlayer, posBox):
    -> we don't need this function due to the nature of the inversed path.
    """
    def isLegalInversion(self, action, posPlayer, posBox): 
        xPlayer, yPlayer = posPlayer
        x1, y1 = xPlayer - action[0], yPlayer - action[1]
        return (x1, y1) not in posBox + self.posWalls and not sorted(posBox) == sorted(self.posGoals)
    
    def legalInvertedUpdate(self, macro, game_data, node):
        player, posBoxes = game_data
        boxes = set(posBoxes)

        for dx, dy in macro:
            new_player = (player[0] - dx, player[1] - dy)

            if new_player in self.posWalls:
                return False, None

            # Determine if a pull move should occur:
            pull_candidate = (player[0] + dx, player[1] + dy)
            pull = pull_candidate in boxes

            if not self.isLegalInversion((dx, dy), player, tuple(boxes)):
                return False, None

            if pull:
                boxes.remove(pull_candidate)
                boxes.add(player)

            player = new_player

        new_state = (player, tuple(boxes))

        new_node = InvertedNode(state=new_state, parent=node, action=macro, rank=node.rank + 1)
        return True, new_node
    
    def initializer(self,initial_grid, end_node):
        '''Iniatilizes the manager with the final grid.'''
        self.posWalls = self.PosOfWalls(initial_grid)
        self.posGoals = self.PosOfGoals(initial_grid)
        end_state = end_node.state
        final_player_pos, final_pos_boxes = end_state[0], end_state[1]
        final_grid = self.final_state_grid(initial_grid, final_player_pos, final_pos_boxes)
        return final_grid
