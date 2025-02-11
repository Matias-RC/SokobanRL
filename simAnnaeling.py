from ast import Tuple
from managers.sokoban_manager import SokobanManager, Node
from collections import defaultdict
import math
import random
import heapq
from typing import Any, Callable, Dict, List, Tuple
from dataclasses import dataclass
from typing import Optional

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

def k_opt(initial_state,final_node,k,moves,manager):
    """
    Perform k-opt analysis on the solution by removing k edges to create sub-tours and reconnect them.
    
    Parameters:
        initial_state: The initial state of the Sokoban game.
        final_node: The final node representing the current solution.
        k: The number of edges to remove.
        moves: List of possible moves (actions).
        manager: A manager instance providing methods like initializer() and LegalUpdate().
        
    Returns:
        The modified final_node after the k-opt improvement.
    """
    _ = manager.initializer(initial_state)
    state_list = final_node.statesList()
    node_list = final_node.nodesList()
    current_node = node_list[0]
    current_index = 0
    
    while current_index + k + 1 < len(node_list):
        frontier = [current_node]
        # Expand the frontier k times using available moves.
        for _ in range(k):
            new_frontier = []
            for node in frontier:
                for move in moves:
                    is_valid, updated_node = manager.LegalUpdate(move, node.state, node)
                    if is_valid:
                        new_frontier.append(updated_node)
            frontier = new_frontier
        
        # Gather candidate nodes that can reconnect to a later state.
        node_proposals: Dict[int, List[Any]] = defaultdict(list)
        for candidate_node in frontier:
            for offset, state in enumerate(state_list[current_index + k + 1:]):
                if candidate_node.state == state:
                    outreach = k + offset + 1
                    node_proposals[outreach].append(candidate_node)
        
        if node_proposals:
            max_outreach = max(node_proposals.keys())
            current_node = node_proposals[max_outreach][0]
            current_index += max_outreach
        else:
            current_index += 1
            current_node = node_list[current_index]
    
    posterior_node = node_list[current_index + 1]
    posterior_node.parent = current_node
    return final_node

# =============================================================================
# Perceived Improvability Function
# =============================================================================
def perceived_improvability(node, delta_scorer, manager, move_library):
    """
    Evaluate potential improvements along a node's trajectory by considering alternative moves.
    
    For each state (except the last two) in the trajectory, compute the baseline quality score.
    Then, for each alternative move from the move_library, compute an alternative quality score.
    If the alternative score is higher, record the improvement details.
    
    Parameters:
        node: The current node with a trajectory.
        delta_scorer: An instance of DeltaScorer to score states.
        manager: A manager instance that provides a LegalUpdate() method.
        move_library: A list of alternative moves (macros) to consider.
        
    Returns:
        A tuple with:
            - The best improvement delta (score difference).
            - A list of improvement details, each containing:
              [baseline_score, improved_score, move, action_sequence, current_state, new_state, index]
    """
    state_list = node.statesList()
    action_sequence = node.trajectory()
    improvements = []
    
    for idx, current_state in enumerate(state_list[:-2]):
        baseline_score = delta_scorer.m(state_list[idx + 1], action_sequence[:idx + 1])
        for macro_move in move_library:
            is_valid, new_node = manager.LegalUpdate(macro=macro_move, game_data=current_state, node=None)
            if is_valid and state_list[idx + 1] != new_node.state:
                # Create a new action sequence by adding the macro_move.
                new_actions = action_sequence[:idx] + [macro_move]
                improved_score = delta_scorer.m(new_node.state, new_actions)
                if baseline_score < improved_score:
                    improvements.append([
                        baseline_score,
                        improved_score,
                        macro_move,
                        action_sequence[:idx],
                        current_state,
                        new_node.state,
                        idx
                    ])
    
    if not improvements:
        return 0, improvements
    else:
        best_delta = max(improvement[1] - improvement[0] for improvement in improvements)
        return best_delta, improvements


def alternative_generator(current_solution, delta_scorer, num_alternatives, improvements, manager, move_library, max_depth=100):
    """
    Generate alternative solution trajectories.
    
    For each candidate node in the improvement bundle:
      - Starting from the candidate node, simulate a trajectory.
      - At each step, gather all legal moves (with resulting nodes) from the current node.
      - Call delta_scorer.q(current_node, legal_moves) to obtain probabilities for the legal actions.
      - Choose the action with the highest probability and update the trajectory.
      - Stop if the state is failed (manager.isFailed) or successful (manager.isEndState),
        or if a maximum depth is reached.
    
    Parameters:
      current_solution: The current solution node (provides nodesList()).
      delta_scorer: An instance with a method q(node, legal_actions) that returns a probability for each move.
      num_alternatives: The number of alternative trajectories to generate.
      improvements: A list of improvement details; each improvement is assumed to have its last element as an index.
      manager: Provides methods such as LegalUpdate(move, state, node), isFailed(node), and isEndState(node).
      max_depth: The maximum depth (number of moves) to simulate for each alternative trajectory.
      move_library: The list of moves to try at each step.
    
    Returns:
      A list of alternative trajectories. Each trajectory is a list of nodes representing a candidate solution.
    """
    # Get the list of nodes from the current solution.
    node_list = current_solution.nodesList()
    
    # Build the initial bundle from improvements.
    # Each improvement is assumed to have its last element as an index into node_list.
    bundle = [(node_list[imp[-1]], imp[0] - imp[1]) for imp in improvements]
    
    # If we have more candidates than needed, keep only the top ones.
    if len(bundle) > num_alternatives:
        bundle.sort(key=lambda x: x[1], reverse=True)
        bundle = bundle[:num_alternatives]
    
    # If the bundle is too short, add random nodes from the solution (ensuring no duplicates).
    used_indices = {imp[-1] for imp in improvements}
    available_indices = list(set(range(len(node_list))) - used_indices)
    while len(bundle) < num_alternatives and available_indices:
        random_index = random.choice(available_indices)
        bundle.append((node_list[random_index], 0))  # Use 0 as a placeholder improvement value.
        available_indices.remove(random_index)
    
    alternative_trajectories = []
    
    # For each candidate node in the bundle, simulate an alternative trajectory.
    for candidate_node, predictedDeltaScore in bundle:
        trajectory = [candidate_node]
        current_node = candidate_node
        current_move_sequence = []  # To track the moves taken in the simulation
        
        for depth in range(max_depth):
            legal_actions = []  # List to store legal moves
            legal_new_nodes = []  # Corresponding new nodes for the legal moves
            
            # Gather all legal moves from the current node.
            for move in move_library:
                is_valid, new_node = manager.LegalUpdate(move, current_node.state, current_node)
                if is_valid:
                    legal_actions.append(move)
                    legal_new_nodes.append(new_node)
            
            # If no legal moves are available, break the simulation.
            if not legal_actions:
                break
            
            # Use the scorer's q method to get probabilities for each legal move.
            action_probabilities = delta_scorer.q(current_node, legal_actions)
            
            # Find the move with the highest probability.
            best_index = None
            best_probability = -1
            for i, prob in enumerate(action_probabilities):
                if prob > best_probability:
                    best_probability = prob
                    best_index = i
            
            # If for some reason no move is selected, break.
            if best_index is None:
                break
            
            best_move = legal_actions[best_index]
            best_new_node = legal_new_nodes[best_index]
            
            # Update the trajectory and move sequence.
            trajectory.append(best_new_node)
            current_move_sequence.append(best_move)
            current_node = best_new_node
            
            # Check terminal conditions.
            if manager.isFailed(current_node):
                break
            if manager.isEndState(current_node):
                break
        
        alternative_trajectories.append((trajectory, predictedDeltaScore))
    
    return alternative_trajectories
    

# =============================================================================
# Simulated Annealing Trajectory Function
# =============================================================================
def simulated_annealing_trajectory(initial_solution,grid,manager,move_library,alternative_generator_fn,delta_scorer,perceived_improvability_fn,num_alternatives = 5,initial_temperature = 1000.0,cooling_rate = 0.95,minimum_temperature= 1e-3):
    """
    Perform simulated annealing on solution trajectories to explore improvements.
    
    The algorithm selects a candidate from the pool, evaluates its potential improvements, generates
    alternative trajectories, and then accepts new trajectories based on an annealing probability.
    
    Parameters:
        initial_solution: The starting solution node.
        manager: The manager instance for state updates.
        move_library: A library of moves (macros) for potential improvements.
        alternative_generator_fn: Function to generate alternative candidate nodes.
        delta_scorer: An instance of DeltaScorer used for scoring.
        perceived_improvability_fn: Function to assess a node's improvability.
        num_alternatives: Number of alternative candidates to generate at each step.
        initial_temperature: Starting temperature for annealing.
        cooling_rate: Factor to decrease the temperature each iteration.
        minimum_temperature: Temperature threshold to stop annealing.
        
    Returns:
        A tuple with:
            - A cache list of candidate solution nodes.
            - Data collected during annealing (improvement details and generated alternatives).
    """
    solution_cache: List[Any] = [initial_solution]
    candidate_pool: List[Any] = [initial_solution]
    temperature = initial_temperature
    annealing_data = []
    # Initialize the manager with the grid.
    manager.initializer(grid)
    # Get grid dimensions.
    grid_shape = grid.shape
    
    while temperature > minimum_temperature and candidate_pool:
        best_candidate = None
        best_improvement_details = None
        best_improvement_score = 0
        
        # Evaluate each candidate in the pool.
        for candidate in candidate_pool:
            improvement_score, improvement_details = perceived_improvability_fn(
                candidate, delta_scorer, manager, move_library
            )
            if improvement_score > best_improvement_score:
                best_improvement_score = improvement_score
                best_candidate = candidate
                best_improvement_details = improvement_details
        
        if best_candidate is None:
            break
        
        candidate_pool.remove(best_candidate)
        
        # Generate alternative trajectories from the best candidate.
        alternative_candidates = alternative_generator_fn(
            best_candidate, delta_scorer, num_alternatives, best_improvement_details, manager, actions_for_sokoban
        )
        annealing_data.append((best_improvement_details, alternative_candidates))
        
        # Decide whether to accept each alternative candidate.
        for alternative, _ in alternative_candidates:
            alternative_quality = len(alternative.statesList())
            candidate_quality = len(best_candidate.statesList())
            
            if alternative_quality < candidate_quality:
                candidate_pool.append(alternative)
                break
            else:
                quality_delta = alternative_quality - candidate_quality
                acceptance_probability = math.exp(-quality_delta / temperature)
                if random.random() < acceptance_probability:
                    candidate_pool.append(alternative)
            solution_cache.append(alternative)
        
        temperature *= cooling_rate
        break
    
    return solution_cache, annealing_data
