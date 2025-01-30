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
    def update_environment(self, grid, action):
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
    """
        Fast Logic --- code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    def PosOfPlayer(self, gameState):
        return tuple(np.argwhere(gameState == 2)[0])

    def PosOfBoxes(self, gameState):
        return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5)))

    def PosOfWalls(self, gameState):
        return tuple(tuple(x) for x in np.argwhere(gameState == 1))

    def PosOfGoals(self, gameState):
        return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5) | (gameState == 6)))

    def isEndState(self, posBox, posGoals):
        return sorted(posBox) == sorted(posGoals)
    
    def isLegalAction(self, action, posPlayer, posBox, posWalls):
        """Check if the given action is legal"""
        xPlayer, yPlayer = posPlayer
        if action[-1].isupper(): # the move was a push
            x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
        else:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]
        return (x1, y1) not in posBox + posWalls

    def legalActions(self, posPlayer, posBox, posWalls):
        """Return all legal actions for the agent in the current game state"""
        allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
        xPlayer, yPlayer = posPlayer
        legalActions = []
        for action in allActions:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]
            if (x1, y1) in posBox: # the move was a push
                action.pop(2) # drop the little letter
            else:
                action.pop(3) # drop the upper letter
            if self.isLegalAction(action, posPlayer, posBox, posWalls):
                legalActions.append(action)
            else:
                continue
        return tuple(tuple(x) for x in legalActions)
    def fastUpdate(self, posPlayer, posBox, action):
        xPlayer, yPlayer = posPlayer # the previous position of player
        newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
        posBox = [list(x) for x in posBox]
        if action[-1].isupper(): # if pushing, update the position of box
            posBox.remove(newPosPlayer)
            posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
        posBox = tuple(tuple(x) for x in posBox)
        newPosPlayer = tuple(newPosPlayer)
        return newPosPlayer, posBox
    def isFailed(self, posBox, posGoals, posWalls):
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
            if box not in posGoals:
                board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                        (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                        (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
                for pattern in allPattern:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                    elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                    elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                    elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                    elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
        return False
    
    """
    InversedLogic
    """

    def inverse_update_environment(self, grid, action):
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
    """
    Fast Inverse logic
    """
    def isLegalInversion(self, action, posPlayer, posBox, posWalls,):
        xPlayer, yPlayer = posPlayer
        x1, y1 = xPlayer - action[0], yPlayer - action[1]
        return (x1, y1) not in posBox + posWalls
    def legalInverts(self, posPlayer, posBox, posWalls, posGoals):
        allActions = [[-1,0,],[1,0],[0,-1],[0,1]]
        #up, down, left, right
        xPlayer, yPlayer = posPlayer
        legalActions = []
        for action in allActions:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]
            temp_boxes = posBox.copy()
            temp_boxes = [(xPlayer, yPlayer) if i == (x1, y1) else i for i in temp_boxes]
            if self.isLegalAction(action, posPlayer, posBox, posWalls) and not self.isEndState(temp_boxes, posGoals):
                legalActions.append(action)
            else:
                continue
        return tuple(tuple(x) for x in legalActions)