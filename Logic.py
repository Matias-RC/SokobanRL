import numpy as np

# Actions mapped to integers
ACTION_MAP = {
    0: (-1, 0),  # 'w' (UP)
    1: (1, 0),   # 's' (DOWN)
    2: (0, -1),  # 'a' (LEFT)
    3: (0, 1)    # 'd' (RIGHT)
}

class Logic():
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
        allActions = [[-1,0], [1,0], [0,-1], [0,1]]  # up, down, left, right
        xPlayer, yPlayer = posPlayer
        legalActions = []
        nextBoxArrengements = []
        for action in allActions:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]

            # Convert tuple to list for modification
            temp_boxes = list(posBox)  
            temp_boxes = [(xPlayer, yPlayer) if i == (x1, y1) else i for i in temp_boxes]

            # Convert back to tuple
            temp_boxes = tuple(temp_boxes)

            if self.isLegalInversion(action, posPlayer, posBox, posWalls) and not self.isEndState(temp_boxes, posGoals):
                legalActions.append(action)
                nextBoxArrengements.append(temp_boxes)
                

        return tuple(tuple(x) for x in legalActions), nextBoxArrengements
    def FastInvert(self, posPlayer, action):
        xPlayer, yPlayer = posPlayer # the previous position of player
        newPosPlayer = [xPlayer - action[0], yPlayer - action[1]] # the current position of player
        return newPosPlayer
    def MoveUntilMultipleOptions(self, posPlayer, posBox, posGoals, posWalls):
        """
        Moves the player (and boxes if needed) until multiple legal inversion actions are available.
        This is a simple iterative approach that stops when more than one inversion is legal.
        Returns:
            (stop_flag, newPosBox, newPosPlayer)
        """
        max_iter = 50  # prevent infinite loops
        iter_count = 0
        while iter_count < max_iter:
            legal_inverts, NewposBox = self.legalInverts(posPlayer, posBox, posWalls, posGoals)
            if len(legal_inverts) > 1:
                return False, posBox, posPlayer
            # If only one move is available, take it.
            if len(legal_inverts) == 0:
                return True, posBox, posPlayer  # stuck
            action = legal_inverts[0]
            posPlayer = self.fastUpdate(posPlayer, posBox, action)
            posBox = NewposBox
            iter_count += 1
        return True, posBox, posPlayer
    def aStar(self, beginPlayer, beginBox, posGoals, posWalls, PriorityQueue, heuristic, cost):
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
            if self.isEndState(node[-1][1], posGoals):
                solution = ','.join(node_action[1:]).replace(',','')
                oneHot = []
                for i in solution:
                    if i == "u" or "U": oneHot.append([1,0,0,0])
                    elif i == "l" or "L": oneHot.append([0,1,0,0])
                    elif i == "d" or "D": oneHot.append([0,0,1,0])
                    else: oneHot.append([0,0,0,1])
                return oneHot
                # break
            if node[-1] not in exploredSet:
                exploredSet.add(node[-1])
                Cost = cost(node_action[1:])
                for action in self.legalActions(node[-1][0], node[-1][1], posWalls):
                    newPosPlayer, newPosBox = self.fastUpdate(node[-1][0], node[-1][1], action)
                    if self.isFailed(newPosBox, posGoals, posWalls):
                        continue
                    count = count + 1
                    Heuristic = heuristic(newPosPlayer, newPosBox, posGoals)
                    frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost)
                    actions.push(node_action + [action[-1]], Heuristic + Cost)