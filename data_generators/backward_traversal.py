import torch
from torch.utils.data import Dataset
import numpy as np
import random
from managers.inverse_manager import InvertedNode




class BackwardTraversalDataset(Dataset):
    def __init__(self, dataset,one_batch = True):
        
        if one_batch:
            self.dataset = [example for example in dataset]
        else:
            self.dataset = [example for batch in dataset for example in batch]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        game_grid, start_state, sub_path_actions, end_state, probability = self.dataset[idx]

        game_grid_tensor = torch.tensor(game_grid, dtype=torch.float32)  
        start_state_tensor = torch.tensor(start_state, dtype=torch.float32)
        sub_path_actions_tensor = torch.tensor(sub_path_actions, dtype=torch.long)  
        end_state_tensor = torch.tensor(end_state, dtype=torch.float32)
        probability_tensor = torch.tensor(probability, dtype=torch.float32)

        return game_grid_tensor, start_state_tensor, sub_path_actions_tensor, end_state_tensor, probability_tensor




class BackwardTraversal:
    def __init__(self,session,model,manager, solver ,maximumDepth=4,maximumBreadth=10,testsPerSearch=None,inverseManager=None, bacthSize = 4, drawSize = 2):
        self.session  = session
        self.model =  model
        self.manager = manager
        self.solver = solver
        self.maxDepth = maximumDepth
        self.maxBreadth = maximumBreadth
        self.cadence = testsPerSearch
        self.inverseManager = inverseManager
        self.batchSize = bacthSize
        self.drawSize = drawSize
        self.datasets = []

    def do(self, session, model):

        for task in session: #recall a session is a set of tasks
            end_node = task.solution
            states_solution, action_solution = end_node.statesList(), end_node.trajectory()
            terminal, initialState = states_solution[-1], states_solution[0]
            #final_grid_state = self.inverseManager.initializer(initial_grid=task.initial_state,end_node=end_node)
            backwards_paths = self.backward_traversal_paths(end_node=end_node,
                                                            initial_grid=task.initial_state,)
            
            for initial_node_path in backwards_paths:
                batch = self.generate_batch(initial_node_path,task=task)
                batch_dataset_torch = BackwardTraversalDataset(batch,one_batch=True)
                self.datasets.append(batch_dataset_torch)
        
        return self.datasets
    """
    def generate_examples(self,initial_node_path,task,n_examples=5):

        batch = []
        for _ in range(n_examples):
            nodes_path = initial_node_path.nodesList()
            complete_path = initial_node_path.statesList()
            actions_path = initial_node_path.trajectory()#[0:-1] #inversed_actions
            positions_path = range(len(complete_path)) #ranking from 0 to len of the path
            rnd_path_indexes = self.get_random_subpath(positions_path)

            start_state, end_state = complete_path[rnd_path_indexes[0]],complete_path[rnd_path_indexes[-1]]
            sub_path_actions = [actions_path[k] for k in rnd_path_indexes]#[0:-1]
            subpath_end_node = nodes_path[rnd_path_indexes[-1]]

            all_paths = self.backward_traversal_all_paths(end_node=subpath_end_node,
                                                            initial_grid=task.initial_state, #map, it does not matter if is the iniital map or the final map
                                                            max_depth=len(rnd_path_indexes),
                                                            max_breadth=10000000000000000000)

            game_grid = self.inverseManager.final_state_grid(initial_grid = task.initial_state,
                                                        final_player_pos=start_state[0],
                                                        final_pos_boxes=start_state[1])

            example = (game_grid,start_state,sub_path_actions,end_state, 1/len(all_paths) ) #map, start_state, actions, end_state, probability
            batch.append(example)
        return batch
    """
    def generate_batch(self,initial_node_path,task,n_examples=5, max_depth=8):
        """
        Different from generate_examples, this function creates a batch of examples from the terminal node
        you retrieve all paths that have a ceratin depth l and then make the batch
        Returns:
            A batch of examples where each   example is  a pair all_paths[i].state, all_paths[i].parent.state
        """
        batch = []
        nodes_path = initial_node_path.nodesList()
        complete_path = initial_node_path.statesList()
        actions_path = initial_node_path.trajectory()

        all_paths = self.backward_traversal_all_paths(end_node=initial_node_path,
                                                        initial_grid=task.initial_state,
                                                        max_depth=max_depth,
                                                        max_breadth=10000000000000000000)
        #Want to create a batch of  (parent, parent_grid, [children], [children_grid])
        node_parents = set()
        for node in all_paths:
            if node.parent is not None:
                node_parents.add(node.parent)
        for node in node_parents:
            game_grid = self.inverseManager.final_state_grid(initial_grid = task.initial_state,
                                                        final_player_pos=node.state[0],
                                                        final_pos_boxes=node.state[1])
            children = [child for child in node.children]
            children_grid = [self.inverseManager.final_state_grid(initial_grid = task.initial_state,
                                                        final_player_pos=child.state[0],
                                                        final_pos_boxes=child.state[1]) for child in children]
            example = (game_grid, node.state, children, children_grid)
            batch.append(example)
        return batch     



    def get_random_subpath(self, indexes_path):
        path_length = len(indexes_path)
        subpath_len = random.randint(2, path_length)  

        start_idx = random.randint(0, path_length - subpath_len)
        subpath_indexes = indexes_path[start_idx : (start_idx + subpath_len)]
        return subpath_indexes

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
    
    def backward_traversal_paths(self,end_node,initial_grid, macros):
        '''Generates the backward traversal paths starting from end node.'''
        final_grid_state = self.inverseManager.initializer(initial_grid=initial_grid,end_node=end_node)
        end_node = InvertedNode(state=end_node.state,parent=None,action=None,inversed_action=None,rank=0)
        frontier = []
        frontier.append(end_node)
        seen_states = set()

        while len(frontier) > 0 and self.maxDepth > 0:
            if len(frontier) < self.maxBreadth:
                new_frontier = []
                
                for node in frontier:
                    if node.state not in seen_states:
                        seen_states.add(node.state)
                        position_player, position_boxes = node.state
                        for m in macros:
                            condition, new_node = self.inverseManager.legalInvertedUpdate(macro=m,
                                                                                        game_data=(position_player,position_boxes),
                                                                                        node=node)
                            if condition:
                                new_frontier.append(new_node)
                max_depth -= 1
                frontier = new_frontier
            else:
                batches = self.makeBatches(frontier, self.batchSize)
                selected_nodes = []

                for batch in batches:
                    states_batch = [node.state for node in batch]
                    q_values = random.choices(range(len(states_batch)),k=len(states_batch))
                    probs = self.GenerateProbs(q_values)
                    selected_indices = random.choices(range(len(batch)), weights=probs, k=self.drawSize)
                    for idx in selected_indices:
                        selected_nodes.append(batch[idx])
                
                frontier = selected_nodes

        print("Number of paths:",len(frontier))
        return frontier
    

    def backward_traversal_all_paths(self, macros, end_node,initial_grid,max_depth,max_breadth=1000000000):
        '''Generates all possible backward traversal paths starting from end node.'''
        
        final_grid_state = self.inverseManager.initializer(initial_grid=initial_grid,end_node=end_node)
        end_node = InvertedNode(state=end_node.state,parent=None,action=None,inversed_action=None,rank=0)
        frontier = []
        frontier.append(end_node)
                
        seen_states = set()
        while 0 < len(self.frontier) < max_breadth and max_depth > 0:
            new_frontier = []
            
            for node in frontier:
                seen_states.add(node.state)
                position_player, position_boxes = node.state
                for m in macros:
                    condition, new_node = self.inverseManager.legalInvertedUpdate(macro=m,
                                                                                game_data=(position_player,position_boxes),
                                                                                node=node)
                    if condition and  new_node.state not in seen_states:
                        node.children.append(new_node)
                        new_frontier.append(new_node)
            max_depth -= 1
            frontier = new_frontier

        print("Number of paths:",len(self.frontier))
        return frontier #self.frontier