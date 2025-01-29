import math
import random
import numpy as np
import torch
import torch.nn as nn
import collections
from Logic import master

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


class PriorityQueue:
    """
    Define a PriorityQueue data structure that will be used
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    def  __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        PriorityQueue.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = PriorityQueue.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

    # Code taken from heapq
    @staticmethod
    def heappush(heap, item):
        """Push item onto heap, maintaining the heap invariant."""
        heap.append(item)
        PriorityQueue._siftdown(heap, 0, len(heap)-1)

    @staticmethod
    def heappop(heap):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            PriorityQueue._siftup(heap, 0)
            return returnitem
        return lastelt

    @staticmethod
    def _siftup(heap, pos):
        endpos = len(heap)
        startpos = pos
        newitem = heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not heap[childpos] < heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = newitem
        PriorityQueue._siftdown(heap, startpos, pos)

    @staticmethod
    def _siftdown(heap, startpos, pos):
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem < parent:
                heap[pos] = parent
                pos = parentpos
                continue
            break
        heap[pos] = newitem
        """Load puzzles and define the rules of sokoban"""

def breadth_first_search(grid, Logic):
    """
    Implement breadthFirstSearch approach
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    beginBox = Logic.PosOfBoxes(grid)
    beginPlayer = Logic.PosOfPlayer(grid)

    startState = (beginPlayer, beginBox)  # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]])  # store states
    actions = collections.deque([[0]])  # store actions
    exploredSet = set()
    count = 0

    posGoals = Logic.PosOfGoals(grid)
    posWalls = Logic.PosOfWalls(grid)

    while frontier:
        node = frontier.popleft()
        node_action = actions.popleft()

        if Logic.isEndState(node[-1][1], posGoals):
            solution = ','.join(node_action[1:]).replace(',', '')
            print(count)
            return solution

        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in Logic.legalActions(node[-1][0], node[-1][1], posWalls):
                count += 1
                newPosPlayer, newPosBox = Logic.fastUpdate(node[-1][0], node[-1][1], action)
                
                if Logic.isFailed(newPosBox, posGoals, posWalls):
                    continue
                
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

def depthFirstSearch(grid, Logic):
    """
    Implement depthFirstSearch approach
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    beginBox = Logic.PosOfBoxes(grid)
    beginPlayer = Logic.PosOfPlayer(grid)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]]
    count = 0

    posGoals = Logic.PosOfGoals(grid)
    posWalls = Logic.PosOfWalls(grid)
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if Logic.isEndState(node[-1][1], posGoals):
            # print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            print(count)
            return solution
            # break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in Logic.legalActions(node[-1][0], node[-1][1], posWalls):
                count = count + 1
                newPosPlayer, newPosBox = Logic.fastUpdate(node[-1][0], node[-1][1], action)
                if Logic.isFailed(newPosBox, posGoals, posWalls):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

def heuristic(posPlayer, posBox, posGoals):
    """
    A heuristic function to calculate the overall distance between the else boxes and the else goals
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance

def cost(actions):
    """
    A cost function
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    return len([x for x in actions if x.islower()])

def uniformCostSearch(grid, Logic):
    """
    Implement uniformCostSearch approach
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    beginBox = Logic.PosOfBoxes(grid)
    beginPlayer = Logic.PosOfPlayer(grid)

    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([startState], 0)
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], 0)
    count = 0

    posGoals = Logic.PosOfGoals(grid)
    posWalls = Logic.PosOfWalls(grid)
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if Logic.isEndState(node[-1][1], posGoals):
            # print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            print(count)
            return solution
            # break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])
            for action in Logic.legalActions(node[-1][0], node[-1][1], posWalls):
                count = count + 1
                newPosPlayer, newPosBox = Logic.fastUpdate(node[-1][0], node[-1][1], action)
                if Logic.isFailed(newPosBox, posGoals, posWalls):
                    continue
                frontier.push(node + [(newPosPlayer, newPosBox)], Cost)
                actions.push(node_action + [action[-1]], Cost)

def aStarSearch(grid, Logic):
    """
    Implement aStarSearch approach
    code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    beginBox = Logic.PosOfBoxes(grid)
    beginPlayer = Logic.PosOfPlayer(grid)
    posGoals = Logic.PosOfGoals(grid)
    posWalls = Logic.PosOfWalls(grid)
    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox, posGoals))
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1], posGoals))
    count = 0
    while frontier:
        # count = count+1
        # print('frontier',frontier)
        if frontier.isEmpty():
            return 'x'
        node = frontier.pop()
        node_action = actions.pop()
        if Logic.isEndState(node[-1][1], posGoals):
            solution = ','.join(node_action[1:]).replace(',','')
            print(count)
            return solution
            # break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])
            for action in Logic.legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = Logic.fastUpdate(node[-1][0], node[-1][1], action)
                if Logic.isFailed(newPosBox, posGoals, posWalls):
                    continue
                count = count + 1
                Heuristic = heuristic(newPosPlayer, newPosBox, posGoals)
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost)
                actions.push(node_action + [action[-1]], Heuristic + Cost)
"""
Example usage:
Easygrid = np.asarray([
    [1,1,1,1,1,1,1],
    [1,0,0,3,4,0,1],
    [1,2,0,3,0,4,1],
    [1,0,1,1,1,1,1],
    [1,0,0,3,0,4,1],
    [1,1,1,1,1,1,1]
])
print(aStarSearch(Easygrid, master()))
"""

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



def ac(actor, critic, episodes, master, max_steps=10, lr_a=1e-3, lr_c=1e-3):
    return

