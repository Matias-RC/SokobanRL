import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from Logic import master

def rotate_matrix(matrix, times):
    return np.rot90(matrix, -times)

def rotate_variations(variations, times):
    """Rotate the variations according to the number of 90-degree clockwise rotations."""
    for _ in range(times):
        variations = [(y, -x) for x, y in variations]
    return variations

def place_template(grid, template, position):

    core, variations = template
    cy, cx = position

    # Place the 3x3 core matrix
    for y in range(core.shape[0]):
        for x in range(core.shape[1]):
            if core[y, x] != -1:  # Avoid overwriting walls
                grid[cy + y, cx + x] = core[y, x]

    # Place variations, respecting grid bounds
    for vy, vx in variations:
        ny, nx = cy + vy + 1, cx + vx + 1  # Adjust relative positions to absolute
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            if grid[ny, nx] == 1:  # Avoid overwriting existing floors or walls
                grid[ny, nx] = 0  # Floors indicated by 0

def construct_grid(level_width, level_height, templates):
    """
    Construct a grid with given dimensions and templates.
    First Step In the overall Generation of a Sokoban Level
    """
    grid = np.ones((level_height, level_width), dtype=int)  # Initialize grid with walls

    # Template dimensions
    template_size = 3

    # Calculate the number of sections based on template size
    grid_height_sections = (level_height + template_size - 1) // template_size
    grid_width_sections = (level_width + template_size - 1) // template_size

    for gy in range(grid_height_sections):
        for gx in range(grid_width_sections):
            # Randomly select and rotate a template
            template = random.choice(templates)
            rotation = random.randint(0, 3)
            core = rotate_matrix(template[0], rotation)
            variations = rotate_variations(template[1], rotation)

            # Determine position to place the 3x3 core matrix, adjusted for boundaries
            cy = min(gy * template_size, level_height - template_size)
            cx = min(gx * template_size, level_width - template_size)
            place_template(grid, [core, variations], (cy, cx))

    return grid

def optimize_grid(grid):
    """Optimize the grid by removing redundant spaces and ensuring accessibility."""
    # Remove isolated floor tiles (surrounded by three or more walls)
    height, width = grid.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y, x] == 0:  # Floor tile
                wall_count = sum(
                    grid[y + dy, x + dx] == 1
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if wall_count >= 3:
                    grid[y, x] = 1  # Convert to wall

    return grid

def is_connected(grid):
    """Check if all floor tiles (0) are connected in the grid."""
    visited = np.zeros_like(grid, dtype=bool)
    start_points = np.argwhere(grid == 0)  # Find all floor tiles
    if len(start_points) == 0:
        return False

    stack = [tuple(start_points[0])]
    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and not visited[ny, nx] and grid[ny, nx] == 0:
                stack.append((ny, nx))

    # Check if all floor tiles are visited
    floor_tiles = (grid == 0)
    return np.all(visited[floor_tiles])

def BuildRoom(level_width, level_height, templates):
    while True:
        grid = construct_grid(level_width, level_height, templates)
        grid = optimize_grid(grid)
        if is_connected(grid):
            break
    return grid

def GenerateEmptyGrid(height, width):
    array = np.zeros((height, width), dtype=int)
    array = np.pad(array, pad_width=1, mode='constant', constant_values=1)
    return array

def FillWithGoalBoxes(grid, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    height, width = grid.shape
    empty_positions = [(i, j) for i in range(1, height - 1) for j in range(1, width - 1)]
    
    np.random.shuffle(empty_positions)
    placed = 0
    for i, j in empty_positions:
        if placed >= n:
            break
        if all(grid[i + di, j + dj] == 0 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            grid[i, j] = 5
            placed += 1
    
    return grid

def FillWithGoalsThenBoxes(emptyPositions, grid, n,seed=None):
    if seed is not None:
        np.random.seed(seed)
    placed =  0
    np.random.shuffle(emptyPositions)
    for i, j in emptyPositions:
        if placed >= n:
            break
        if all(grid[i + di, j + dj] == 0 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            grid[i, j] = 4
            emptyPositions = np.argwhere(grid == 0)
            placed += 1    
    placed = 0
    np.random.shuffle(emptyPositions)
    for i, j in emptyPositions:
        if placed >= n:
            break
        if all(grid[i + di, j + dj] == 0 or grid[i + di, j + dj] == 4 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            grid[i, j] = 3
            placed += 1  
    return grid
    

def FillWithWalls(grid, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    height, width = grid.shape
    empty_positions = [(i, j) for i in range(1, height - 1) for j in range(1, width - 1) if grid[i, j] == 0]
    
    np.random.shuffle(empty_positions)
    placed = 0
    for i, j in empty_positions:
        if placed >= n:
            break
        temp_grid = grid.copy()
        temp_grid[i, j] = 1
        
        fives_positions = np.argwhere(temp_grid == 5)
        if all(any(temp_grid[i + di, j + dj] == 0 for di, dj in [(-1, 0), (0, 1)]) for i, j in fives_positions) and all(any(temp_grid[i+di,j+dj] == 0 for di, dj in [(1, 0), (0, -1)]) for i, j in fives_positions) and is_connected(temp_grid):
            grid[i, j] = 1
            placed += 1
    
    return grid

def PlacePlayer(grid, seed=None):
    if seed is not None:
        np.random.seed(seed)
    posibleEndStates = []
    goalBoxes = np.argwhere(grid == 5)
    for i, j in goalBoxes:
        for di, dj in [(-1, 0),(1, 0), (0, -1),(0, 1)]:
            try:
                if grid[i+di,j+dj] == 0 and grid[i+di*2,j+dj*2] == 0:
                    posibleEndStates.append((i+di,j+dj))
            except: 
                pass
                
    np.random.shuffle(posibleEndStates)
    try:
        y,x = posibleEndStates[0]
        grid[y,x] = 2
        return grid
    except:
        return False
def RandomPlacePlayer(empty_positions, grid, seed=None):
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(empty_positions)
    grid[empty_positions[0][0],empty_positions[0][1]] = 2
    return grid


"""
----------------------------------------------------------------------
More utils:
----------------------------------------------------------------------
"""

def MakeSeedsList(n):
    return [random.randint(0, 10000) for _ in range(n)]

def MakeDimsList(rango, reference, n):
    return [(random.randint(reference-rango, reference+rango),
             random.randint(reference-rango, reference+rango)) for _ in range(n)]

def RandVariablelist(rango, reference, n):
    return [random.randint(reference-rango,reference+rango) for _ in range(n)]

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
    Logic.PosGoals = posGoals
    Logic.PosWalls = posWalls

    while frontier:
        node = frontier.popleft()
        node_action = actions.popleft()

        if Logic.isEndState(node[-1][1]):
            solution = node_action[1:]
            print(count)
            return solution

        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in Logic.legalActions(node[-1][0], node[-1][1]):
                count += 1
                newPosPlayer, newPosBox = Logic.fastUpdate(node[-1][0], node[-1][1], action)
                
                if Logic.isFailed(newPosBox):
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
            solution = node_action[1:]
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
    return len([x for x in actions if x[-1] == 0])

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
            solution = node_action[1:]
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
            solution = node_action[1:]
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


#Example usage:
Easygrid = np.asarray([
    [1,1,1,1,1,1,1],
    [1,0,0,3,4,0,1],
    [1,2,0,3,0,4,1],
    [1,0,1,1,1,1,1],
    [1,0,0,3,0,4,1],
    [1,1,1,1,1,1,1]
])
print(breadth_first_search(Easygrid, master(heuristic, cost)))

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