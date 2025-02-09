from managers.sokoban_manager import SokobanManager, Node
from collections import defaultdict
import math
import random
import heapq
from typing import Callable, List, Any
from dataclasses import dataclass
from typing import Optional

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

def kOpt(initial_state, terminalNode, k,l, manager):
    """
    k-opt analysis involves deleting k edges from the current solution to the problem, 
    creating k sub-tours. -> combinatorial
    """
    _ = manager.initializer(initial_state)
    statesList = terminalNode.statesList()
    nodesList = terminalNode.nodesList()
    node = nodesList[0]
    idx  = 0
    while idx + k+1< len(nodesList):
        frontier = [node]
        for _ in range(k):
            new_frontier = []
            for i in frontier:
                for action in l:
                    bool_condition, new_node = manager.LegalUpdate(action,i.state, i)
                    if bool_condition:
                        new_frontier.append(new_node)
            frontier = new_frontier

        nodeProposals = defaultdict(list)
        outreach = 1  # Default outreach

        for i in frontier:
            for j, s in enumerate(statesList[idx + k + 1:]):
                if i.state == s:
                    outreach = k + j + 1
                    nodeProposals[outreach].append(i)  # Store the proposal correctly
        if nodeProposals.keys():  # This ensures all lists have at least one valid proposal
            biggest_outreach = max(nodeProposals.keys())  # Get the highest outreach key
            node = nodeProposals[biggest_outreach][0]  # Get the first proposal with the biggest outreach
            idx += biggest_outreach 
        else:
            idx += 1
            node = nodesList[idx]
            
    posterior = nodesList[idx+1]
    posterior.parent = node
    return terminalNode

def simulated_annealing_trajectory(
    initial_solution: Node,
    generator: Callable[[Node], List[Node]],
    q: Callable[[Node], float],
        # neighbor_generator takes a current solution (Node) and returns a list of new candidate solutions.
    T_init: float = 1000.0,
    alpha: float = 0.95,
    T_min: float = 1e-3,
    max_iter: int = 1000,
) -> Node:   
    current = initial_solution.statesList()
    iteration = 0
    T = T_init

    while iteration < max_iter and T > T_min:
        altTrajectories = generator(current, q)
        if not altTrajectories:
            break

        current_quality = len(current)
        
        improved = [len(t.statesList()) for t in altTrajectories if len(t.statesList()) < current_quality]

        if improved:
            candidate = min(improved)
            current = candidate
        else:
            candidates = [t for t in altTrajectories]
            candidate_length = float("inf")
            candidate = None
            for t in candidates:
                if len(t.statesList()) < candidate_length:
                    candidate_length = len(t.statesList())
                    candidate = t

            delta = candidate_length - current_quality 
            if random.random() < math.exp(-delta / T):
                current = candidate

        T *= alpha
        iteration += 1

    return current