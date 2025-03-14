import numpy as np
from collections import defaultdict
import random

# Actions mapped to integers
ACTION_MAP = {
    0: (-1, 0),  # 'w' (UP)
    1: (1, 0),   # 's' (DOWN)
    2: (0, -1),  # 'a' (LEFT)
    3: (0, 1)    # 'd' (RIGHT)
}
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



class master():
    def __init__(self, heuristic, cost):
        self.solution_cache = {}
        self.visited_states = defaultdict(int)
        self.non_determinism_factor = 3
        # Add these required attributes
        self.posWalls = None
        self.posGoals = None
        self.heuristic = heuristic  # From earlier code
        self.cost = cost  # From earlier code
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
    
    def isLegalAction(self, action, posPlayer, posBox):
        """Check if the given action is legal"""
        xPlayer, yPlayer = posPlayer
        if action[-1][-1] == 1: # the move was a push
            x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
        else:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]
        return (x1, y1) not in posBox + self.posWalls

    def legalActions(self, posPlayer, posBox):
        """Return all legal actions for the agent in the current game state"""
        allActions = [[-1,0,[1,0,0,0,0],[1,0,0,0,1]],
                      [0,-1,[0,1,0,0,0],[0,1,0,0,1]],
                      [1,0,[0,0,1,0,0],[0,0,1,0,1]],
                      [0,1,[0,0,0,1,0],[0,0,0,1,1]]]
        xPlayer, yPlayer = posPlayer
        legalActions = []
        for action in allActions:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]
            if (x1, y1) in posBox: # the move was a push
                action.pop(2) # drop the little letter
            else:
                action.pop(3) # drop the upper letter
            if self.isLegalAction(action, posPlayer, posBox):
                legalActions.append(action)
            else:
                continue
        return tuple(tuple(x) for x in legalActions)
    def fastUpdate(self, posPlayer, posBox, action):
        xPlayer, yPlayer = posPlayer # the previous position of player
        newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
        newPosBox = [list(x) for x in posBox]
        
        if action[-1][-1] == 1: # if pushing, update the position of box
            newPosBox.remove(newPosPlayer)
            newPosBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
        newPosBox = tuple(tuple(x) for x in newPosBox)
        newPosPlayer = tuple(newPosPlayer)
        return newPosPlayer, newPosBox
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
    
    """
    InversedLogic
    """

    def inverse_update_environment(self, grid, action):
        """
        Reverses the Sokoban environment based on the action and modifies the grid in place.
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
    def isLegalInversion(self, action, posPlayer, posBox):
        xPlayer, yPlayer = posPlayer
        x1, y1 = xPlayer - action[0], yPlayer - action[1]
        return (x1, y1) not in posBox + self.posWalls
    def legalInverts(self, posPlayer, posBox):
        allActions = [(-1,0), (0,-1), (1,0), (0,1)]
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

            if self.isLegalInversion(action, posPlayer, posBox) and not self.isEndState(temp_boxes):
                legalActions.append(action)
                nextBoxArrengements.append(temp_boxes)
                
        return tuple(tuple(x) for x in legalActions), nextBoxArrengements
    def FastInvert(self, posPlayer, action):
        xPlayer, yPlayer = posPlayer # the previous position of player
        newPosPlayer = (xPlayer - action[0], yPlayer - action[1]) # the current position of player
        return newPosPlayer
    def MoveUntilMultipleOptions(self, posPlayer, posBox):
        """
        Moves the player (and boxes if needed) until multiple legal inversion actions are available.
        This is a simple iterative approach that stops when more than one inversion is legal.
        Returns:
            (stop_flag, newPosBox, newPosPlayer)
        """
        max_iter = 50  # prevent infinite loops
        iter_count = 0
        while iter_count < max_iter:
            legal_inverts, NewposBox = self.legalInverts(posPlayer, posBox)
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
    def aStar(self, beginPlayer, beginBox):
        start_state = (beginPlayer, beginBox)
        frontier = PriorityQueue()
        frontier.push([start_state], self.heuristic(beginPlayer, beginBox, self.posGoals))
        exploredSet = set()
        actions = PriorityQueue()
        actions.push([0], self.heuristic(beginPlayer, start_state[1], self.posGoals))
        count = 0
        while frontier:
            # count = count+1
            # print('frontier',frontier)
            if frontier.isEmpty():
                return 'x'
            node = frontier.pop()
            node_action = actions.pop()
            if self.isEndState(node[-1][1]):
                solution = node_action[1:]
                return solution
                # break
            if node[-1] not in exploredSet:
                exploredSet.add(node[-1])
                Cost = self.cost(node_action[1:])
                for action in self.legalActions(node[-1][0], node[-1][1]):
                    newPosPlayer, newPosBox = self.fastUpdate(node[-1][0], node[-1][1], action)
                    if self.isFailed(newPosBox):
                        continue
                    count = count + 1
                    Heuristic = self.heuristic(newPosPlayer, newPosBox, self.posGoals)
                    frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost)
                    actions.push(node_action + [action[-1]], Heuristic + Cost)


    def calculate_box_lines(self, solution, player_pos, box_pos):
        """Improved with proper action parsing"""
        if not solution or solution == 'x':
            return 0

        current_dir = None
        box_lines = 0

        for action in solution:
            # Convert numeric action to direction
            if action == 0:  # up
                new_dir = (-1, 0)
            elif action == 1:  # down
                new_dir = (1, 0)
            elif action == 2:  # left
                new_dir = (0, -1)
            elif action == 3:  # right
                new_dir = (0, 1)
            else:
                continue

            # Check if push action
            is_push = self._is_push_action(action, player_pos, box_pos)

            if is_push:
                if new_dir != current_dir:
                    box_lines += 1
                    current_dir = new_dir
            else:
                current_dir = None

        return box_lines

    def state_heuristic(self, player_pos, box_pos):
        """Combined heuristic using cached A* solution properties"""
        state_key = (player_pos, tuple(sorted(box_pos)))
        
        if state_key in self.solution_cache:
            solution, length, lines = self.solution_cache[state_key]
            return length*.7 + lines  # Weight box lines metric
        
        # Compute and cache if not exists
        solution = self.aStar(player_pos, box_pos, self.posWalls, self.posGoals, 
                             self.heuristic, self.cost)
        if solution == 'x':
            return float('inf')  # Unsolvable
            
        length = len(solution)
        lines = self.calculate_box_lines(solution)
        self.solution_cache[state_key] = (solution, length, lines)
        
        return length*.7 + lines
    
    # --- Companion function: Generate a probability distribution over leaf scores ---
    @staticmethod
    def GenerateProbDistributionForLeafs(scores):
        """
        Given a list of scores (where a lower score is better), returns a list of indices
        representing the selected subset of leaves (approximately one fourth of the total).
        The selection is probabilistic with better (i.e. lower) scores getting higher probability.
        """
        n = len(scores)
        if n == 0:
            return []
        k = max(1, n // 4)  # select at least one leaf
        epsilon = 1e-6
        # Use inverse cost (lower cost gets higher weight)
        weights = [1.0 / (s + epsilon) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]
        # Sample indices (allowing duplicates) then remove duplicates.
        selected_indices = random.choices(range(n), weights=probs, k=k)
        return list(set(selected_indices))
    
    # --- The Depth and Breadth Limited Search using inversion moves ---
    def DepthAndBreadthLimitedSearch(self, posPlayer, posBox, max_depth, max_breadth):
        """
        Performs a search that alternates between full expansion (when the breadth is small)
        and probabilistic pruning (when the breadth is large). It expands inversion moves up
        to max_depth levels. At the end, it returns the state (player and box positions) with the
        best (lowest) heuristic value, along with its cached solution.
        """
        # Start with the initial state as the only leaf.
        leafs = [(posPlayer, posBox)]
        depth = max_depth
        while depth > 0:
            
            # Expand without pruning if the number of leafs is small.
            if len(leafs) < max_breadth:
                new_leafs = []
                for state in leafs:
                    current_player, current_box = state
                    legal_inverts, next_box_arrangements = self.legalInverts(current_player, current_box)
                    for i, action in enumerate(legal_inverts):
                        new_player = self.FastInvert(current_player, action)
                        new_box = next_box_arrangements[i]
                        state_key = (new_player, tuple(sorted(new_box)))
                        if state_key not in self.solution_cache:
                            sol = self.aStar(new_player, new_box)
                            lines = self.calculate_box_lines(sol, new_player, new_box)
                            self.solution_cache[state_key] = sol, len(sol), lines
                        new_leafs.append((new_player, new_box))
                leafs = new_leafs
                depth -= 1
            else:
                # When breadth is high, compute heuristic scores and prune a portion of the leaves.
                scores = []
                for state in leafs:
                    player, box = state
                    h_val = self.state_heuristic(player, box)
                    scores.append(h_val)
                selected_indices = self.GenerateProbDistributionForLeafs(scores)
                pruned_leafs = [leafs[i] for i in selected_indices]
                new_leafs = []
                for state in pruned_leafs:
                    current_player, current_box = state
                    legal_inverts, next_box_arrangements = self.legalInverts(current_player, current_box)
                    for i, action in enumerate(legal_inverts):
                        new_player = self.FastInvert(current_player, action)
                        new_box = next_box_arrangements[i]
                        state_key = (new_player, tuple(sorted(new_box)))
                        if state_key not in self.solution_cache:
                            sol = self.aStar(new_player, new_box)
                            lines = self.calculate_box_lines(sol, new_player, new_box)
                            self.solution_cache[state_key] = sol, len(sol), lines
                        new_leafs.append((new_player, new_box))
                leafs = new_leafs
                depth -= 1
    def get_longest_solution_from_cache(self):
        worst_state_key = None
        worst_solution = None
        worst_score = -float('inf')

        for state_key, (solution, length, lines) in self.solution_cache.items():
            if solution == 'x':
                continue
            score = length + lines * 0.5

            if score > worst_score:
                worst_score = score
                worst_state_key = state_key
                worst_solution = solution

        if worst_state_key is None:
            return None  # No valid solution found.

        return worst_state_key, worst_solution, worst_score