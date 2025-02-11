import random
from data.task_solver_interaction import Task_Solver_Interaction
from managers.sokoban_manager import SokobanManager
from dcLogic import Solver #solver for sokoban
import numpy as np

class MonteCarloTreeSearch:
    def __init__(self,library_actions,manager,batchSize,drawSize,max_depth=100,max_breadth=10000):
        self.manager = manager
        self.library_actions = library_actions
        self.frontier = []
        self.max_depth = max_depth
        self.max_breadth = max_breadth

        self.batchSize = batchSize
        self.drawSize = drawSize

        self.seen_states = set() # O(1) for search  

        self.is_first_session = True

        Keep_register = True
        
    def GenerateProbs(self,values):
        if not values:
            return []
        values = np.array(values, dtype=np.float64)
        exp_values = np.exp(values - np.max(values)) 
        probabilities = exp_values / np.sum(exp_values) # Normalize to sum to 1
        return probabilities.tolist()

    def makeBatches(self, nodesList, batchSize):
        n = len(nodesList)
        numBatches = n // batchSize

        if n % batchSize != 0:
            numBatches += 1

        # Randomly shuffle indices to ensure randomness
        indices = list(range(n))
        random.shuffle(indices)

        batches = [[] for _ in range(numBatches)]
        for i, idx in enumerate(indices):
            batch_index = i % numBatches 
            batches[batch_index].append(nodesList[idx])
        return batches
    
    def do(self,task,q_function):
        node = self.manager.initializer(task.initial_state)
        self.frontier.append(node)
        depth = self.max_depth
        while len(self.frontier) >  0 and depth > 0:
            if len(self.frontier) < self.max_breadth:
                new_frontier = []
                for node in self.frontier:
                    if node.state not in self.seen_states:
                        self.seen_states.add(node.state)
                        for action in self.library_actions:
                            bool_condition, new_node = self.manager.LegalUpdate(macro=action,game_data=node.state,node=node)
                            if bool_condition:
                                if self.manager.isEndState(node=new_node):
                                    self.frontier = []
                                    self.seen_states = set()
                                    return new_node #solution
                                new_frontier.append(new_node)
                        
                self.frontier = new_frontier
                depth -= 1
            else:
                batches = self.makeBatches(self.frontier, self.batchSize)
                selected_nodes = []

                for batch in batches:
                    states_batch = [node.state for node in batch]
                    q_values = q_function(states_batch)
                    probs = self.GenerateProbs(q_values)
                    selected_indices = random.choices(range(len(batch)), weights=probs, k=self.drawSize)
                    for idx in selected_indices:
                        selected_nodes.append(batch[idx])
                        
                self.frontier = selected_nodes
                depth -= 1
        self.seen_states = set()
        
    def do_first_session(self,task,q_function):
        pass
    
    def do_next_sessions(self,task,q_function):
        pass

                    