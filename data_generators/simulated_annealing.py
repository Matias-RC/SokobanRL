from ast import Tuple
from managers.sokoban_manager import SokobanManager, Node
from collections import defaultdict
import math
import random
import heapq
from typing import Any, Callable, Dict, List, Tuple
from dataclasses import dataclass
from typing import Optional


class simulatedAnnealing:
    def __init__(self, session, manager, deepRankModel, solver, library):
        self.manager = manager
        self.deepRankModel = deepRankModel
        self.session = session
        self.solver = solver
        self.library = library

    def node_exists_in_pool(new_node, pool):
        return any(existing_node.state == new_node.state for existing_node in pool)
    
    def annealing(self, initial_solution, grid, primitives, initial_temperature=1000.0, cooling_rate=0.95, minimum_temperature=1e-3):
        """
        Performs a simulated annealing process to refine an initial solution, represented as a node at a terminal state.  
        The goal is to iteratively improve the trajectory by exploring neighboring nodes while balancing exploration and exploitation.

        The procedure follows these steps:
        1. Extract all nodes from the trajectory of the initial solution.
        2. Initialize a pool containing these trajectory nodes.
        3. Identify all direct neighbors of the nodes in the pool.
        4. Evaluate the neighboring nodes using the deepRankModel.
        5. Select a neighbor probabilistically using a softmax-like scheme where candidates with a higher rank (score) 
           are more likely to be chosen.
        6. Set the selected neighbor as the starting point for a new trajectory.
        7. Generate a new trajectory using self.solver(neighbor).
        8. Accept the new trajectory based on the acceptance criterion:  
           - Accept if exp(-Δcost/temperature) > random(0,1), where Δcost is the difference in trajectory cost.
        9. Expand the pool by adding all new nodes from the accepted trajectory.
        10. Keep track of all explored trajectories for analysis and further optimization.
        """
        # 1. Initialize the pool with all nodes from the initial solution's trajectory.
        pool = list(initial_solution.nodesList())
        temperature = initial_temperature
        trajectories = [initial_solution]
        self.manager.initializer(grid)

        while temperature > minimum_temperature:
            candidate_neighbors = []
            
            # 2. For each node in the pool, generate candidate neighbors.
            for node in pool:
                current_rank = self.deepRankModel.foward(node.state)
                for action in primitives:
                    condition, new_node = self.manager.legalUpdate(node, action)
                    if condition:
                        # Skip if the new_node is already in the pool.
                        if self.node_exists_in_pool(new_node, pool):
                            continue
                        new_rank = self.deepRankModel.foward(new_node.state)
                        improvement = new_rank - current_rank
                        candidate_neighbors.append((new_node, new_rank, improvement))
            
            # If no candidates are found, exit the loop.
            if not candidate_neighbors:
                print("No candidate neighbors found; ending annealing.")
                break

            # 3. Probabilistic neighbor selection using a softmax-like weighting.
            total_weight = sum(math.exp(candidate[1]) for candidate in candidate_neighbors)
            r = random.random() * total_weight
            cumulative = 0.0
            selected_candidate = None
            for candidate in candidate_neighbors:
                weight = math.exp(candidate[1])
                cumulative += weight
                if cumulative >= r:
                    selected_candidate = candidate
                    break

            # Fallback in case no candidate was selected (should not happen).
            if selected_candidate is None:
                selected_candidate = candidate_neighbors[0]

            # 4. Generate a new trajectory starting from the selected candidate.
            new_trajectory = self.solver.do(selected_candidate[0])
            
            # 5. Compute the cost difference between the new trajectory and the current solution.
            current_solution = trajectories[-1]
            current_cost = len(current_solution.statesList())
            new_cost = len(new_trajectory.statesList())
            delta = new_cost - current_cost  # A negative delta implies an improvement.

            acceptance_probability = math.exp(-delta / temperature) if delta > 0 else 1.0
            if acceptance_probability > random.random():
                trajectories.append(new_trajectory)
                for new_node in new_trajectory.nodesList():
                    if self.node_exists_in_pool(new_node, pool):
                        pool.append(new_node)
                print(f"(delta: {delta}) at temperature {temperature:.4f}.")
            else:
                print(f"(delta: {delta}) at temperature {temperature:.4f}.")

            temperature *= cooling_rate

        return trajectories
    

