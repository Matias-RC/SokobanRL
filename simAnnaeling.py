from managers.sokoban_manager import SokobanManager

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

def kOpt(initial_state, terminalNode, k,l, manager=SokobanManager):
    """
    k-opt analysis involves deleting k edges from the current solution to the problem, 
    creating k sub-tours. -> combinatorial
    """
    node = manager.initializer(manager,initial_state)
    statesList = terminalNode.statesList(terminalNode)

    for idx, state in enumerate(statesList):
        frontier = [node]
        for depth in range(k):
            new_frontier = []
            for i in frontier:
                for action in l:
                    bool_condition, new_node = manager.LegalUpdate(manager, action, statesList[idx], i)
                    if bool_condition:
                        new_frontier.append(new_node)
            frontier = new_frontier
                
            





def simulatedAnnaeling(trajectory):
    pass