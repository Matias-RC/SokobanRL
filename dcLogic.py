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

class Solver():
    def __init__(self, q, L):
        self.cache = {}
        self.posWalls = None
        self.posGoals = None
        self.q = q
        self.L = L
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
    def GenerateProbs(values):
        if not values:
            return []

        values = np.array(values, dtype=np.float64)

        exp_values = np.exp(values - np.max(values)) 

        # Normalize to sum to 1
        probabilities = exp_values / np.sum(exp_values)

        return probabilities.tolist()
    def makeBatches(nodesList, batchSize):
        n = len(nodesList)
        numBatches = n // batchSize

        if n % batchSize != 0:
            numBatches += 1

        # Randomly shuffle indices to ensure randomness
        indices = list(range(n))
        random.shuffle(indices)

        batches = [[] for _ in range(numBatches)]
        for i, idx in enumerate(indices):
            batch_index = i % numBatches 
            batches[batch_index].append(nodesList[idx])
        return batches
    
    def dcSolve(self, posPlayer, posBox, max_depth, max_breadth, batchSize, drawSize):
        root = Node(state=(posPlayer, posBox), parent=None, action=None)
        frontier = Frontier()
        frontier.push(root)

        exploredSet = set()

        while not frontier.is_empty():


            if frontier.size() < max_breadth:
                new_nodes = []
                while not frontier.is_empty():
                    node = frontier.pop()
                    currentState = node.state
                    if self.isEndState(currentState[1]):
                        return node.trajectory()
                    for action in self.L:
                        isLegal, newPosPlayer, newPosBox = self.LegalUpdate(action, currentState[0], currentState[1])
                        if isLegal and (newPosPlayer, newPosBox) not in exploredSet:
                            newNode = Node(state=(newPosPlayer, newPosBox), parent=node, action=action)
                            new_nodes.append(newNode)
                            exploredSet.add((newPosPlayer, newPosBox))
                for node in new_nodes:
                    frontier.push(node)
            else:
                nodesList = list(frontier.frontier)
                batches = self.makeBatches(nodesList, batchSize)
                selected_nodes = []

                for batch in batches:
                    states_batch = [node.state for node in batch]
                    q_values = self.q.predict(states_batch)
                    probs = self.GenerateProbs(q_values)
                    selected_indices = random.choices(range(len(batch)), weights=probs, k=drawSize)
                    for idx in selected_indices:
                        selected_nodes.append(batch[idx])
                frontier.frontier.clear()
                for node in selected_nodes:
                    frontier.push(node)