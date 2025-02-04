import numpy as np
from collections import defaultdict, deque
import random
from dataclasses import dataclass
from typing import Any, Optional, List

InitalLibrary = ([(-1,0)],[(1, 0)],[(0,-1)],[(0, 1)])

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
            path.append(node)
            node = node.parent
        return list(reversed(path))

class Frontier:
    def __init__(self):
        self.frontier = deque()

    def push(self, node: Node):
        """
        Add a new node to the frontier.
        """
        self.frontier.append(node)

    def pop(self) -> Node:
        """
        Remove and return the next node according to the removal policy.
        For BFS, we pop from the left of the deque.
        """
        return self.frontier.popleft()

    def is_empty(self) -> bool:
        """
        Check whether the frontier is empty.
        """
        return not self.frontier

class dreamCoder():
    def __init__(self, q, pi, L, frontier):
        self.cache = {}
        self.posWalls = None
        self.posGoals = None
        self.q = q
        self.pi = pi
        self.L = L
        self.frontier = frontier
    def PosOfPlayer(self, grid):
        return tuple(np.argwhere(grid == 2)[0])

    def PosOfBoxes(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 3) | (grid == 5)))

    def PosOfWalls(self, grid):
        return tuple(tuple(x) for x in np.argwhere(grid == 1))

    def PosOfGoals(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 4) | (grid == 5) | (grid == 6)))

    def isEndState(self, posBox):
        return sorted(posBox) == sorted(self.posGoals)    
    
    def isLegalAction(self, action, posPlayer, posBoxes):
        dx, dy = action[0]
        factor = 2 if action[1] else 1
        target = (posPlayer[0] + factor * dx, posPlayer[1] + factor * dy)
        return target not in self.walls and target not in posBoxes


    def LegalUpdate(self, macro, posPlayer, posBoxes):
        player = posPlayer
        boxes = set(posBoxes)

        for dx, dy in macro:
            nextPos = (player[0] + dx, player[1] + dy)
            push = nextPos in boxes
            action = ((dx, dy), push)
            if not self.isLegalAction(action, player, boxes):
                return False, None, None
            player = nextPos
            if push:
                boxes.remove(player)
                boxes.add((player[0] + dx, player[1] + dy))

        return True, player, tuple(boxes)
    
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
    def GenerateProbs():
        pass
    def dcSolve(self, posPlayer, posBox, max_depth, max_breadth, expantionQuota):
        self.frontier = Frontier()
        depth = max_depth
        if self.isEndState(posBox): return False
        while depth > 0:
           if len(self.frontier) < max_breadth:
               currentPlayer, currentBox = self.frontier.pop().state
               legals = []
               for primitive in self.L:
                   isLegal, newPosPlayer, newPosBox = self.LegalUpdate(primitive, currentPlayer, currentBox)
                   if isLegal:
                       legals.append()
                       

        pass