import numpy as np
import random

def rotate_matrix(matrix, times):
    """Rotate a matrix 90 degrees clockwise a given number of times."""
    return np.rot90(matrix, -times)

def rotate_variations(variations, times):
    """Rotate the variations according to the number of 90-degree clockwise rotations."""
    for _ in range(times):
        variations = [(y, -x) for x, y in variations]
    return variations

def place_template(grid, template, position):
    """Place a template's core and variations on the grid."""
    core, variations = template
    cy, cx = position
    
    # Place the 3x3 core matrix
    for y in range(core.shape[0]):
        for x in range(core.shape[1]):
            grid[cy + y, cx + x] = core[y, x]

    # Place variations, respecting grid bounds
    for vy, vx in variations:
        ny, nx = cy + vy + 1, cx + vx + 1  # Adjust relative positions to absolute
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            grid[ny, nx] = 0  # Floors indicated by 0

def construct_grid(level_width, level_height, templates):
    """Construct a grid with given dimensions and templates."""
    # Ensure dimensions are divisible by 3
    if level_width % 3 != 0 or level_height % 3 != 0:
        raise ValueError("Both level_width and level_height must be divisible by 3.")

    grid = np.ones((level_height, level_width), dtype=int)  # Initialize grid with walls
    grid_height_sections = level_height // 3
    grid_width_sections = level_width // 3

    for gy in range(grid_height_sections):
        for gx in range(grid_width_sections):
            # Randomly select and rotate a template
            template = random.choice(templates)
            rotation = random.randint(0, 3)
            core = rotate_matrix(template[0], rotation)
            variations = rotate_variations(template[1], rotation)

            # Determine position to place the 3x3 core matrix
            cy, cx = gy * 3, gx * 3
            place_template(grid, [core, variations], (cy, cx))

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
