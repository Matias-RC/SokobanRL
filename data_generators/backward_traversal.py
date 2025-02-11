from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from managers.inverse_manager import InversedSokobanManager
from managers.inverse_manager import Node


class BackwardTraversal:
    def __init__(self,session=None,model=None,manager=None,maximumDepth=10,maximumBreadth=3,testsPerSearch=None,inverseManager=None, bacthSize = 4, drawSize = 1):
        self.session  = session
        self.model =  model
        self.manager = manager
        self.maxDepth = maximumDepth
        self.maxBreadth = maximumBreadth
        self.cadence = testsPerSearch
        self.inverseManager = inverseManager
        self.batchSize = bacthSize
        self.drawSize = drawSize
    
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
            backwards_paths = self.backward_traversal_paths(end_node=end_node,
                                                            initial_grid=task.initial_state,
                                                            max_depth=self.maxDepth,
                                                            max_breadth=self.maxBreadth,)
            print("OK")
            dataset.append(self.generate())
        return dataset

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
    
    def backward_traversal_paths(self,end_node,initial_grid,max_depth,max_breadth):
        '''Generates the backward traversal paths starting from end node.'''
        final_grid_state = self.inverseManager.initializer(initial_grid=initial_grid,end_node=end_node)

        self.frontier = []
        self.frontier.append(end_node)
        self.seen_states = set()

        while len(self.frontier) > 0 and max_depth > 0:
            if len(self.frontier) < max_breadth:
                new_frontier = []
                for node in self.frontier:
                    if node.state not in self.seen_states:
                        self.seen_states.add(node.state)
                        position_player, position_boxes = node.state
                        bool_condition, legalActions_boxArrengements = self.inverseManager.legalInverts(posPlayer=position_player,posBox=position_boxes) 
                        legalActions, boxArrengements = legalActions_boxArrengements
                        if bool_condition:
                            for action,box_arr in zip(legalActions,boxArrengements):
                                new_node = Node(state=(self.inverseManager.FastInvert(position_player,action),box_arr),parent=node,action=action)
                                if not self.inverseManager.isEndState(new_node): #if the state is not the end state then save, whe need to move boxes
                                    new_frontier.append(new_node)
                self.maxDepth -= 1
                self.frontier = new_frontier
            else:
                batches = self.makeBatches(self.frontier, self.batchSize)
                selected_nodes = []

                for batch in batches:
                    states_batch = [node.state for node in batch]
                    q_values = random.choices(range(len(states_batch)),k=len(states_batch))
                    probs = self.GenerateProbs(q_values)
                    selected_indices = random.choices(range(len(batch)), weights=probs, k=self.drawSize)
                    for idx in selected_indices:
                        selected_nodes.append(batch[idx])
                        
                self.frontier = selected_nodes
        print("OK")