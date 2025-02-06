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
    _ = manager.initializer(manager,initial_state)
    statesList = terminalNode.statesList(terminalNode)
    nodesList = terminalNode.nodesList(terminalNode)
    node = nodesList[0]
    idx  = 0
    while idx + k < len(nodesList):
        frontier = [node]
        for _ in range(k):
            new_frontier = []
            for i in frontier:
                for action in l:
                    bool_condition, new_node = manager.LegalUpdate(manager, action,i.state, i)
                    if bool_condition:
                        new_frontier.append(new_node)
            frontier = new_frontier
        nodeProposals = defaultdict(list)
        outreach = 1
        for i in frontier:
            proposal = None
            for j, s in enumerate(statesList[idx+k+1:]):
                if i.state == s:
                    outreach = k + j + 1
                    proposal = i
            nodeProposals[outreach].append(proposal)
        biggest_outreach = max(nodeProposals)
        pnode = nodeProposals[biggest_outreach][0]
        idx += outreach
        if pnode is not None:
            node = pnode
        else:
            node = nodesList[idx]
    posterior = nodesList[idx]
    posterior.parent = node
    return terminalNode

def simulatedAnnaeling(trajectory):
    pass