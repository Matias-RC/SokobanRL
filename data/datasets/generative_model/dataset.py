import torch
from torch.utils.data import Dataset
import numpy as np
import random

class GenerativeDataset(Dataset):

    '''GenerativeDataset recieves task and generate grid(x), actions(y) pairs'''
    
    def __init__(self, session_batch):
        super().__init__()  # Ensures compatibility with torch Dataset
        
        # grid (initial state) and the position of the sokoban player
        initial_states = [s.initial_state for s in session_batch]
        # actions that should generate the model
        actions_to_solve = [sol.solution.trajectory() for sol in session_batch]  
        print(actions_to_solve)
        #print(initial_states)
        print(actions_to_solve,len(actions_to_solve))

        self.dataset = {"initial_states": initial_states ,
                        "actions_to_solve":actions_to_solve, }

    
    def contruct_examples(self, batch):
        n = len(batch)
        indices = list(range(n))
        random.shuffle(indices)   

        examples = [batch[i] for i in indices]

        return examples
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        x_grid = torch.tensor(self.dataset["initial_states"][idx] , dtype=torch.long)
        y_actions = torch.tensor(self.dataset["actions_to_solve"][idx], dtype=torch.long)
        
        return {
            "encoder_x": x_grid,  
            "decoder_x": x_grid,  
            "y_actions": y_actions,  
            "shape": x_grid.shape[0]
        }