from data.task import Task
import numpy as np
from collections import defaultdict, deque
import random
from dataclasses import dataclass
from typing import Any, Optional, List

@dataclass
class Node:
    state: Any
    parent: Optional['Node'] = None
    action: Optional[Any] = None

    def trajectory(self) -> List['Node']:
        """
        Reconstructs the trajectory (path) from the root to this node.
        """
        node, path = self, []

        while node:
            path.append(node.action)
            node = node.parent
            
        return list(reversed(path))[1:]
    
    def statesList(self) -> List['Node']:
        node, path = self, []

        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
    
    def nodesList(self) -> List['Node']:
        node, path = self, []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
class SokobanManager:
    def __init__(self):
        self.posWalls = None
        self.posGoals = None

    def PosOfPlayer(self, grid):
        return tuple(np.argwhere(grid == 2)[0]) # idplayer = 2

    def PosOfBoxes(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 3) | (grid == 5)))

    def PosOfWalls(self, grid):
        return tuple(tuple(x) for x in np.argwhere(grid == 1))

    def PosOfGoals(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 4) | (grid == 5) | (grid == 6)))

    def isEndState(self,node):
        return sorted(node.state[1]) == sorted(self.posGoals)    
    
    def isLegalAction(self, action, posPlayer, posBoxes):
        dx, dy = action[0]
        factor = 2 if action[1] else 1
        target = (posPlayer[0] + factor * dx, posPlayer[1] + factor * dy)
        return target not in self.posWalls and target not in posBoxes
    
    def isFailed(self, posBox):
        """This function used to observe if the state is potentially failed, then prune the search"""
        rotatePattern = [[0,1,2,3,4,5,6,7,8],
                        [2,5,8,1,4,7,0,3,6],
                        [0,1,2,3,4,5,6,7,8][::-1],
                        [2,5,8,1,4,7,0,3,6][::-1]]
        flipPattern = [[2,1,0,5,4,3,8,7,6],
                        [0,3,6,1,4,7,2,5,8],
                        [2,1,0,5,4,3,8,7,6][::-1],
                        [0,3,6,1,4,7,2,5,8][::-1]]
        allPattern = rotatePattern + flipPattern

        for box in posBox:
            if box not in self.posGoals:
                board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                        (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                        (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
                for pattern in allPattern:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in self.posWalls and newBoard[5] in self.posWalls: return True
                    elif newBoard[1] in posBox and newBoard[2] in self.posWalls and newBoard[5] in self.posWalls: return True
                    elif newBoard[1] in posBox and newBoard[2] in self.posWalls and newBoard[5] in posBox: return True
                    elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                    elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in self.posWalls and newBoard[3] in self.posWalls and newBoard[8] in self.posWalls: return True
        return False

    def LegalUpdate(self,macro,game_data,node): #posPlayer, posBoxes
        player, posBoxes = game_data
        boxes = set(posBoxes)

        for dx, dy in macro:
            nextPos = (player[0] + dx, player[1] + dy)
            push = nextPos in boxes
            action = ((dx, dy), push)
            if not self.isLegalAction(action, player, boxes):
                return False, None
            player = nextPos
            if push:
                boxes.remove(player)
                boxes.add((player[0] + dx, player[1] + dy))
        posBoxes = tuple(boxes)
        new_node = Node(state=(player,posBoxes),parent=node,action=macro)
        condition = not self.isFailed(posBoxes)
        return condition, new_node
    
    def initializer(self,initial_state):
        self.posWalls = self.PosOfWalls(initial_state)
        self.posGoals = self.PosOfGoals(initial_state)
        node = Node(state=(self.PosOfPlayer(initial_state), self.PosOfBoxes(initial_state)))
        return node