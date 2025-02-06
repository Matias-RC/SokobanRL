from managers.sokoban_manager import SokobanManager
from collections import defaultdict

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

def simulatedAnnaeling(trajectory):
    pass