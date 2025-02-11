from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class BackwardTraversal:
    def __init__(self,session=None,model=None,manager=None,maximumDepth=10,maximumBreadth=3,testsPerSearch=None,inverseManager=None):
        self.session  = session
        self.model =  model
        self.manager = manager
        self.maxDepth = maximumDepth
        self.maxBreadth = maximumBreadth
        self.cadence = testsPerSearch
        self.inverseManager = inverseManager
    
    def generate_examples(self,paths):
        for path in paths:
            #select random initial and terminal nodes
            init_node = random.choice(path)
            term_node = random.choice(path[init_node.pos:])
            example = (init_node,term_node,init_node.trajectory(term_node)) #nodes and actions
        batch = []
        
    def do(self, session, model):
        dataset = []
        for task in session: #recall a session is a set of tasks
            end_node = task.solution
            states_solution, action_solution = end_node.statesList(), end_node.trajectory()
            terminal, initialState = states_solution[-1], states_solution[0]
            #final_grid_state = self.inverseManager.initializer(initial_grid=task.initial_state,end_node=end_node)
            backwards_paths = self.inverseManager.backward_traversal_paths(end_node=end_node,
                                                            initial_grid=task.initial_state,
                                                            max_depth=self.maxDepth,
                                                            max_breadth=self.maxBreadth,)
            print("OK")
            dataset.append(self.generate())
        return dataset