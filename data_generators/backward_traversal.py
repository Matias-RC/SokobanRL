import torch
from torch.utils.data import Dataset
import numpy as np
import random
from managers.inverse_manager import InvertedNode
from SokoSource  import PriorityQueue
import math
from data.datasets.backward_traversal.dataset import BackwardTraversalDataset


class BackwardTraversal:#
    def __init__(self,
                 session,
                 model,
                 manager,
                 agent,           
                 maximumDepth=4,
                 maximumBreadth=10,
                 testsPerSearch=None,
                 inverseManager=None,
                 batchSize=4,     
                 drawSize=2):
        
        self.session = session
        self.model = model
        self.manager = manager
        self.agent = agent  
        self.maxDepth = maximumDepth
        self.maxBreadth = maximumBreadth
        self.cadence = testsPerSearch
        self.inverseManager = inverseManager
        self.batchSize = batchSize
        self.drawSize = drawSize

        self.datasets = []

    def do(self, session, model):
        for task in session:  # a session is a set of tasks
            end_node = task.solution
            states_solution, _ = end_node.statesList(), end_node.trajectory()
            terminal, initialState = states_solution[-1], states_solution[0]
            batch = self.generate_batch(end_node, task=task)
            batch_dataset_torch = BackwardTraversalDataset(batch)
            self.datasets.append(batch_dataset_torch)
        
        return self.datasets

    def generate_batch(self, initial_node_path, task, max_depth=8):
        """
        Creates a batch of examples from the terminal node.
        Returns:
            A batch of tuples with the following structure: (game_grid, node.rank)
        """ 
        batch = []

        frontier, end_node = self.backward_traversal_all_paths(
                                    end_node=initial_node_path,
                                    initial_grid=task.initial_state,
                                    max_depth=max_depth,
                                    max_breadth=self.maxBreadth)
        # end_node has children; we want to make the batch from all its children
        childs = end_node.children
        while childs:
            new_childs = []
            for node in childs:
                game_grid = self.inverseManager.final_state_grid(
                                initial_grid=task.initial_state,
                                final_player_pos=node.state[0],
                                final_pos_boxes=node.state[1])
                example = {"grid": game_grid, "rank": node.rank}
                batch.append(example)
                if node.children:
                    new_childs.extend(node.children)
            childs = new_childs
        return batch

    def get_random_subpath(self, indexes_path):
        path_length = len(indexes_path)
        subpath_len = random.randint(2, path_length)  
        start_idx = random.randint(0, path_length - subpath_len)
        subpath_indexes = indexes_path[start_idx : (start_idx + subpath_len)]
        return subpath_indexes

    def GenerateProbs(self, values):
        if not values:
            return []
        values = np.array(values, dtype=np.float64)
        exp_values = np.exp(values - np.max(values))
        probabilities = exp_values / np.sum(exp_values)  # Normalize to sum to 1
        return probabilities.tolist()
    
    def makeBatches(self, nodesList, batchSize):
        n = len(nodesList)
        numBatches = n // batchSize
        if n % batchSize != 0:
            numBatches += 1
        indices = list(range(n))
        random.shuffle(indices)
        batches = [[] for _ in range(numBatches)]
        for i, idx in enumerate(indices):
            batch_index = i % numBatches 
            batches[batch_index].append(nodesList[idx])
        return batches
    
    def backward_traversal_all_paths(self, end_node, initial_grid, max_depth, max_breadth=1000000000):
        '''Generates all possible backward traversal paths starting from end_node.'''
        
        final_grid_state = self.inverseManager.initializer(initial_grid=initial_grid, end_node=end_node)
        # Create an InvertedNode from end_node.
        end_node = InvertedNode(state=end_node.state, parent=None, action=None, inversed_action=None, rank=0)
        frontier = [end_node]
        seen_states = set()
        
        while 0 < len(frontier) < max_breadth and max_depth > 0:
            new_frontier = []
            for node in frontier:
                seen_states.add(node.state)
                position_player, position_boxes = node.state
                for m in self.agent.library:
                    condition, new_node = self.inverseManager.legalInvertedUpdate(
                        macro=m,
                        game_data=(position_player, position_boxes),
                        node=node)
                    if condition and new_node.state not in seen_states:
                        node.children.append(new_node)
                        new_frontier.append(new_node)
            max_depth -= 1
            frontier = new_frontier

        print("Number of paths:", len(frontier))
        return frontier, end_node 
    
    def backward_traversal_worst_paths(self, end_node, initial_grid, max_depth, max_frontier_capacity=1000000000):
        final_grid_state = self.inverseManager.initializer(initial_grid=initial_grid, end_node=end_node)
        # Create an InvertedNode from end_node.
        end_node = InvertedNode(state=end_node.state, parent=None, action=None, inversed_action=None, rank=0)

        frontier = PriorityQueue()
        frontier.push(end_node, 0)
        depth = max_depth
        seen_states = set()
        while not frontier.isEmpty() and frontier.Count < max_frontier_capacity and depth > 0:
            node = frontier.pop()
            for m in self.agent.library:
                conditition,  new_node = self.inverseManager.legalInvertedUpdate(m,node.state,node)
                if conditition:
                    node.children.append(new_node)
                    frontier.push(new_node,(-self.model(new_node.state)+1)*20+len(new_node.trayectory()))
            depth -= 1
        print("Count:", frontier.Count)
        return frontier, end_node