import numpy as np
import pickle
import SokoSource as source
from Logic import master


#with open("templates/templates.pkl", "rb") as f:
#    templates = pickle.load(f)

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


def FillWithWalls(empty_positions,grid, n, seed=None):
    if seed is not None:
        np.random.seed(seed)

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



def Scenario(height=8,width=8):
        solution = "x"
        while solution == "x":
            #Generate room
            room = GenerateEmptyGrid(width,height) #inner_room
            #room = np.ones((height + 2, width + 2), dtype=inner_room.dtype)
            #room[1:-1, 1:-1] = inner_room
            room = FillWithWalls(np.argwhere(room == 0),room, 4, seed=None)
            room = FillWithGoalsThenBoxes(np.argwhere(room == 0),room, 2, seed=None)
            room = RandomPlacePlayer(np.argwhere(room == 0), room, seed=None)
            logic = master(source.heuristic, source.cost)
            posBox =  logic.PosOfBoxes(room)
            posPlayer = logic.PosOfPlayer(room)
            logic.posGoals = logic.PosOfGoals(room)
            logic.posWalls = logic.PosOfWalls(room)
            solution =  logic.aStar(posPlayer,posBox)
        return room
        





