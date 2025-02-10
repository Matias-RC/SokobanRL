from ast import Tuple
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

def perceivedImprovability(
        node: Node,
        q: Callable[[Tuple[Tuple[int]],List[List[Tuple[int]]]], float], 
        manager:SokobanManager,
        library:List[List[Tuple[int]]]
) -> float:
    states = node.statesList()
    actions = node.trajectory()
    perceivedImprovements = []
    for idx, nodeState in enumerate(states[:-2]):
        nextValue = q(states[idx+1], actions[:idx+1])
        for action in library:
            bool_condition, new_node = manager.LegalUpdate(macro=action,game_data=nodeState,node=None) 
            if bool_condition and states[idx+1] != new_node.state:
                percivedValue = q(new_node.state)
                if nextValue < percivedValue:
                    perceivedImprovements.append([nextValue, percivedValue, action, nodeState, new_node.state])
    if not perceivedImprovements:
        return 0
    else:
        score = 0
        for i in perceivedImprovements:
            delta = i[1]-i[0]
            if score < delta:
                score = delta
        return score


def simulated_annealing_trajectory(
    initial_solution: Node,
    manager:SokobanManager,
    library: List[List[Tuple[int]]],
    generator: Callable[[Node, Callable[[Node], float], int], List[Node]],
    q: Callable[[Tuple[Tuple[int]]], float],
    perceivedImprovability: Callable[[Node], float],
    num_alternatives: int = 5,
    T_init: float = 1000.0,
    alpha: float = 0.95,
    T_min: float = 1e-3
) -> Node:
    cache: List[Node] = [initial_solution]
    pool: List[Node] = [initial_solution]
    T = T_init

    while T > T_min and pool:
        candidate = max(pool, key=lambda node: perceivedImprovability(node, q, manager, library))
        pool.remove(candidate)
        altTrajectories = generator(candidate, q, num_alternatives)


        for alt in altTrajectories:
            alt_quality = len(alt.statesList())
            candidate_quality = len(candidate.statesList())
            if alt_quality < candidate_quality:
                pool.append(alt)
                break
            else:
                delta = alt_quality - candidate_quality
                if random.random() < math.exp(-delta / T):
                    pool.append(alt)
            cache.append(alt)
        T *= alpha

    return cache