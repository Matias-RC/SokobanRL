import numpy as np
import random

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

def GenerateEmpty(height, width):
    array = np.zeros((height, width), dtype=int)
    array = np.pad(array, pad_width=1, mode='constant', constant_values=1)
    return array