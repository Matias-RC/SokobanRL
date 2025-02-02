import numpy as np
from collections import defaultdict
import random

# Define action mappings
ACTION_MAP = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
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
        """Load puzzles and define the rules of sokoban"""
class master:
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
        player_pos = np.argwhere((grid == 2) | (grid == 6))
        if len(player_pos) == 0:
            raise ValueError("Player not found in the grid.")
        player_pos = tuple(player_pos[0])

        row, col = player_pos
        d_row, d_col = ACTION_MAP[action]
        new_row, new_col = row + d_row, col + d_col

        if not (0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]):
            return grid, False

        target_cell = grid[new_row, new_col]

        if target_cell == 0:  # Empty space
            grid[new_row, new_col] = 2
            grid[row, col] = 4 if grid[row, col] == 6 else 0
        elif target_cell == 3:  # Box
            box_new_row, box_new_col = new_row + d_row, new_col + d_col
            if (0 <= box_new_row < grid.shape[0] and 0 <= box_new_col < grid.shape[1] and
                    grid[box_new_row, box_new_col] in [0, 4]):
                grid[box_new_row, box_new_col] = 5 if grid[box_new_row, box_new_col] == 4 else 3
                grid[new_row, new_col] = 2
                grid[row, col] = 4 if grid[row, col] == 6 else 0
        elif target_cell == 4:  # Button
            grid[new_row, new_col] = 6
            grid[row, col] = 4 if grid[row, col] == 6 else 0
        elif target_cell == 5:  # Box on Button
            box_new_row, box_new_col = new_row + d_row, new_col + d_col
            if (0 <= box_new_row < grid.shape[0] and 0 <= box_new_col < grid.shape[1] and
                    grid[box_new_row, box_new_col] in [0, 4]):
                grid[box_new_row, box_new_col] = 5 if grid[box_new_row, box_new_col] == 4 else 3
                grid[new_row, new_col] = 6
                grid[row, col] = 4 if grid[row, col] == 6 else 0

        terminal_success = not (4 in grid or 6 in grid)
        return grid, terminal_success

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
        xPlayer, yPlayer = posPlayer
        if action[-1][-1] == 1:  # Push action
            x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
        else:
            x1, y1 = xPlayer + action[0], yPlayer + action[1]
        return (x1, y1) not in posBox + posWalls

    def legalActions(self, posPlayer, posBox, posWalls):
        allActions = [
            [-1, 0, [1, 0, 0, 0, 0], [1, 0, 0, 0, 1]],  # Up
            [0, -1, [0, 1, 0, 0, 0], [0, 1, 0, 0, 1]],  # Left
            [1, 0, [0, 0, 1, 0, 0], [0, 0, 1, 0, 1]],   # Down
            [0, 1, [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]]    # Right
        ]
        legalActions = []
        for action in allActions:
            x1, y1 = posPlayer[0] + action[0], posPlayer[1] + action[1]
            if (x1, y1) in posBox:  # Push action
                action.pop(2)
            else:
                action.pop(3)
            if self.isLegalAction(action, posPlayer, posBox, posWalls):
                legalActions.append(action)
        return tuple(tuple(x) for x in legalActions)

    def fastUpdate(self, posPlayer, posBox, action):
        xPlayer, yPlayer = posPlayer
        newPosPlayer = [xPlayer + action[0], yPlayer + action[1]]
        posBox = [list(x) for x in posBox]
        if action[-1][-1] == 1:  # Push action
            posBox.remove(newPosPlayer)
            posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
        posBox = tuple(tuple(x) for x in posBox)
        newPosPlayer = tuple(newPosPlayer)
        return newPosPlayer, posBox

    def isFailed(self, posBox, posGoals, posWalls):
        for box in posBox:
            if box not in posGoals:
                board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                         (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                         (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
                for pattern in [[0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 5, 8, 1, 4, 7, 0, 3, 6],
                                [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1], [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1]]:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in posWalls and newBoard[5] in posWalls:
                        return True
        return False

    def calculate_box_lines(self, solution):
        if not solution or solution == 'x':
            return 0
        current_dir = None
        box_lines = 0
        for action in solution:
            if isinstance(action, list) or isinstance(action, tuple):
                if action[-1] == 1:  # Push action
                    dir_idx = action[:-1].index(1)
                    new_dir = ['up', 'down', 'left', 'right'][dir_idx]
                    if new_dir != current_dir:
                        box_lines += 1
                        current_dir = new_dir
                else:
                    current_dir = None
        return box_lines

    def state_heuristic(self, player_pos, box_pos, posGoals):
        state_key = (player_pos, tuple(sorted(box_pos)))
        if state_key in self.solution_cache:
            solution, length, lines = self.solution_cache[state_key]
            return length + lines * 0.5
        solution = self.aStar(player_pos, box_pos, self.posWalls, posGoals, PriorityQueue(), self.heuristic, self.cost)
        if solution == 'x':
            return float('inf')
        length = len(solution)
        lines = self.calculate_box_lines(solution)
        self.solution_cache[state_key] = (solution, length, lines)
        return length + lines * 0.5

    def legalInverts(self, posPlayer, posBox, posWalls, posGoals):
        all_actions = [
            ((-1, 0), [1, 0, 0, 0, 0]),  # Up
            ((0, -1), [0, 1, 0, 0, 0]),  # Left
            ((1, 0), [0, 0, 1, 0, 0]),   # Down
            ((0, 1), [0, 0, 0, 1, 0])    # Right
        ]
        scored_actions = []
        for action, encoding in all_actions:
            new_player = (posPlayer[0] + action[0], posPlayer[1] + action[1])
            if not self.isLegalInversion(action, posPlayer, posBox, posWalls):
                continue
            new_boxes = [b for b in posBox if b != new_player]
            if encoding[-1] == 1:  # Push action
                new_boxes.append(posPlayer)
            score = self.state_heuristic(new_player, tuple(sorted(new_boxes)), posGoals)
            scored_actions.append((-score, (new_player, tuple(sorted(new_boxes))), (action, encoding)))
        scored_actions.sort(key=lambda x: x[0])
        top_candidates = scored_actions[:self.non_determinism_factor]
        if not top_candidates:
            return [], []
        selected = random.choice(top_candidates)
        return [selected[2]], [selected[1][1]]

    def depthLimitedSearch(self, posPlayer, posBox, depth, max_branches=30):
        if depth == 0 or len(self.visited_states) >= max_branches:
            return self.solution_cache.get((posPlayer, tuple(sorted(posBox))), (None, 0, 0))
        legal_actions, new_box_states = self.legalInverts(posPlayer, posBox, self.posWalls, self.posGoals)
        best_solution = None
        best_score = float('inf')
        for action, new_boxes in zip(legal_actions, new_box_states):
            if self.visited_states[(action, new_boxes)] > 2:
                continue
            self.visited_states[(action, new_boxes)] += 1
            solution, length, lines = self.depthLimitedSearch(action[0], new_boxes, depth - 1)
            current_score = length + lines * 0.5
            if current_score < best_score:
                best_score = current_score
                best_solution = solution
        return best_solution, best_score, self.calculate_box_lines(best_solution)

    def isLegalInversion(self, action, posPlayer, posBox, posWalls):
        x1, y1 = posPlayer[0] + action[0], posPlayer[1] + action[1]
        return (x1, y1) not in posBox + posWalls

    def FastInvert(self, posPlayer, action):
        return (posPlayer[0] + action[0][0], posPlayer[1] + action[0][1])

    def analyze_grid(self, grid):
        """Initialize wall and goal positions from the grid."""
        self.posWalls = self.PosOfWalls(grid)
        self.posGoals = self.PosOfGoals(grid)