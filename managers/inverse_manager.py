from turtle import pos
import numpy as np
from collections import defaultdict, deque
import random
from dataclasses import dataclass
from typing import Any, Optional, List



@dataclass
class InvertedNode:
    ''' Data structure for a node in the search tree. '''
    state: Any
    parent: Optional['InvertedNode'] = None
    action: Optional[Any] = None
    inversed_action: Optional[Any] = None
    rank: int = 0

    def trajectory(self) -> List['InvertedNode']:
        node, path = self, []

        while node:
            path.append(node.action)
            node = node.parent
            
        return list(path)#[1:]
    
    def statesList(self) -> List['InvertedNode']:
        node, path = self, []

        while node:
            path.append(node.state)
            node = node.parent
        return list(path)
    
    def nodesList(self) -> List['InvertedNode']:
        node, path = self, []
        while node:
            path.append(node)
            node = node.parent
        return list(path)
    
    def inversed_trajectory(self) -> List['InvertedNode']:
        node, path = self, []

        while node:
            path.append(node.inversed_action)
            node = node.parent
            
        return list(path)


class InversedSokobanManager:
    ''' Manager for sokoban that generate an inversed path.'''
    def __init__(self):
        self.posWalls = None
        self.posGoals = None
    
    def PosOfWalls(self, grid):
        return tuple(tuple(x) for x in np.argwhere(grid == 1))

    def PosOfGoals(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 4) | (grid == 5) | (grid == 6)))
    
    def PosOfBoxes(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 3) | (grid == 5)))
    
    def isEndState(self,node):
        '''Check if the position is winner state.'''
        return sorted(node.state[1]) == sorted(self.posGoals)
        
    def final_state_grid(self, initial_grid, final_player_pos, final_pos_boxes):
        '''Creates the final grid from the initial grid and the final player and box positions.'''
        final_grid = np.copy(initial_grid) #copy
        final_grid[(final_grid == 2) | (final_grid == 3) | (final_grid == 5) | (final_grid == 6)] = 0 #reset
        
        if tuple(final_player_pos) in self.posGoals:
            final_grid[tuple(final_player_pos)] = 6  # Player on Button
        else:
            final_grid[tuple(final_player_pos)] = 2  # Normal Player
        
        for box in final_pos_boxes:
            if tuple(box) in self.posGoals:
                final_grid[tuple(box)] = 5  # Box on Button
            else:
                final_grid[tuple(box)] = 3  # Normal Box
        
        return final_grid
    """
    def isFailed(self, posPlayer, posBox):
    -> we don't need this function due to the nature of the inversed path.
    """
    def isLegalInversion(self, action, posPlayer, posBox): 
        xPlayer, yPlayer = posPlayer
        x1, y1 = xPlayer - action[0], yPlayer - action[1]
        return (x1, y1) not in posBox + self.posWalls
    
    def legalInvertedUpdate(self, macro, game_data, node):
        player, posBoxes = game_data
        boxes = set(posBoxes)

        for dx, dy in macro:
            new_player = (player[0] - dx, player[1] - dy)

            if new_player in self.posWalls:
                return False, None

            # Determine if a pull move should occur:
            pull_candidate = (player[0] + dx, player[1] + dy)
            pull = pull_candidate in boxes

            if not self.isLegalInversion((dx, dy), player, tuple(boxes)):
                return False, None

            if pull:
                boxes.remove(pull_candidate)
                boxes.add(player)

            player = new_player

        new_state = (player, tuple(boxes))

        new_node = InvertedNode(state=new_state, parent=node, action=macro)
        return True, new_node
    
    def initializer(self,initial_grid, end_node):
        '''Iniatilizes the manager with the final grid.'''
        self.posWalls = self.PosOfWalls(initial_grid)
        self.posGoals = self.PosOfGoals(initial_grid)
        end_state = end_node.state
        final_player_pos, final_pos_boxes = end_state[0], end_state[1]
        final_grid = self.final_state_grid(initial_grid, final_player_pos, final_pos_boxes)
        return final_grid
    ################################################################
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
    '''
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
'''