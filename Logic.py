import numpy as np

# Actions mapped to integers
ACTION_MAP = {
    0: (-1, 0),  # 'w' (UP)
    1: (1, 0),   # 's' (DOWN)
    2: (0, -1),  # 'a' (LEFT)
    3: (0, 1)    # 'd' (RIGHT)
}

class master():
    def __init__(self):
        pass
    def update_environment(grid, action):
        """
        Updates the Sokoban environment based on the action and modifies the grid in place.

        Args:
            grid (np.ndarray): A matrix representing the current environment state.
                Legend:
                    0: Empty space
                    1: Wall
                    2: Player
                    3: Box
                    4: Button
                    5: Box on Button
                    6: Player on Button
            action (int): Action to perform (0='w', 1='s', 2='a', 3='d').

        Returns:
            (np.ndarray, bool): Updated grid and a boolean indicating terminal success.
        """
        # Locate player position
        player_pos = np.argwhere((grid == 2) | (grid == 6))
        if len(player_pos) == 0:
            raise ValueError("Player not found in the grid.")
        player_pos = tuple(player_pos[0])  # Extract the first match (row, col)

        # Calculate new player position
        row, col = player_pos
        d_row, d_col = ACTION_MAP[action]
        new_row, new_col = row + d_row, col + d_col

        # Check bounds
        if not (0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]):
            return grid, False  # Invalid move (out of bounds)

        # Check target cell
        target_cell = grid[new_row, new_col]

        # Handle different interactions
        if target_cell == 0:  # Empty space
            grid[new_row, new_col] = 2
            grid[row, col] = 4 if grid[row, col] == 6 else 0
        elif target_cell == 3:  # Box
            # Check the cell beyond the box
            box_new_row, box_new_col = new_row + d_row, new_col + d_col
            if (0 <= box_new_row < grid.shape[0] and 0 <= box_new_col < grid.shape[1] and
                    grid[box_new_row, box_new_col] in [0, 4]):  # Box can move
                grid[box_new_row, box_new_col] = 5 if grid[box_new_row, box_new_col] == 4 else 3
                grid[new_row, new_col] = 2 
                grid[row, col] = 4 if grid[row, col] == 6 else 0
        elif target_cell == 4:  # Button or box on button
            grid[new_row, new_col] = 6
            grid[row, col] = 4 if grid[row, col] == 6 else 0
        elif target_cell == 5:
            # Check the cell beyond the box
            box_new_row, box_new_col = new_row + d_row, new_col + d_col
            if (0 <= box_new_row < grid.shape[0] and 0 <= box_new_col < grid.shape[1] and
                    grid[box_new_row, box_new_col] in [0, 4]):  # Box can move
                grid[box_new_row, box_new_col] = 5 if grid[box_new_row, box_new_col] == 4 else 3
                grid[new_row, new_col] = 6 
                grid[row, col] = 4 if grid[row, col] == 6 else 0

        # Check for terminal success: all buttons are covered by boxes
        terminal_success = not (4 in grid or 6 in grid)

        return grid, terminal_success

    def inverse_update_environment(grid, action):
        """
        Reverses the Sokoban environment based on the action and modifies the grid in place.
    
        Args:
            grid (np.ndarray): A matrix representing the current environment state.
                Legend:
                    0: Empty space
                    1: Wall
                    2: Player
                    3: Box
                    4: Button
                    5: Box on Button
                    6: Player on Button
            action (int): Action to reverse (0='w', 1='s', 2='a', 3='d').
    
        Returns:
            np.ndarray: Updated grid after the inverse action.
        """
        # Locate player position
        player_pos = np.argwhere((grid == 2) | (grid == 6))
        if len(player_pos) == 0:
            raise ValueError("Player not found in the grid.")
        player_pos = tuple(player_pos[0])  # Extract the first match (row, col)
    
        # Calculate the source cell (where the player came from)
        row, col = player_pos
        d_row, d_col = ACTION_MAP[action]
        prev_row, prev_col = row - d_row, col - d_col  # Source cell
        front_row, front_col = row + d_row, col + d_col  # Cell in front of the player
    
        # Check bounds for source and front cells
        if not (0 <= prev_row < grid.shape[0] and 0 <= prev_col < grid.shape[1]):
            return grid  # Invalid move (out of bounds)
    
        if grid[prev_row, prev_col] == 3 or grid[prev_row, prev_col] == 5 or grid[prev_row, prev_col] == 1:
            # Invalid move: Player cannot reverse into a box nor a wall
            return grid
        if grid[front_row, front_col] == 3:
            #pull back
            grid[front_row, front_col] = 0
            grid[row, col] = 5 if grid[row, col] == 6 else 3
            grid[prev_row, prev_col] = 6 if grid[prev_row, prev_col] == 4 else 2
        elif grid[front_row, front_col] == 5:  # Box or box on button
            # Pull the box back
            grid[front_row, front_col] = 4
            grid[row, col] = 5 if grid[row, col] == 6 else 3
            grid[prev_row, prev_col] = 6 if grid[prev_row, prev_col] == 4 else 2
        else:
            # Move player to the source cell
            grid[row, col] = 4 if grid[row, col] == 6 else 0
            grid[prev_row, prev_col] = 6 if grid[prev_row, prev_col] == 4 else 2
        return grid