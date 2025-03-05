import torch
from torch.utils.data import Dataset
import numpy as np
import random

class GenerativeDataset(Dataset):

    '''GenerativeDataset recieves task and generate grid(x), actions(y) pairs'''
    
    def __init__(self,
                 session_batch,
                 samples_per_session=1,
                 dsl={(0,1):0,
                          (1,0):1,
                          (0,-1):2,
                          (-1,0):3},
                 block_size=4 ):
        
        super().__init__()  # Ensures compatibility with torch Dataset
        
        # grid (initial state) and the position of the sokoban player
        initial_states = [s.initial_state for s in session_batch]
    
        # actions that should generate the model in order to solve the game
        self.states=[]
        self.x=[]
        self.y=[]
        self.all_actions=[]
        
        actions_to_solve = [ sol.solution.trajectory() for sol in session_batch]
        actions_encoded   = [ [dsl[a[0]] for a in k] for k in actions_to_solve ]  
        
        for i in range(len(actions_encoded)):

            actions_seq = actions_encoded[i]
            num_actions = len(actions_seq)
            
            idx = np.random.choice(a=(num_actions-block_size),size=samples_per_session) 

            input_dec = [actions_seq[ k : (k+block_size)] for k in idx] 
            output_dec = [actions_seq[ (k+1):(k+block_size+1)] for k in idx]

            self.all_actions.append(actions_seq)
            self.x.extend(input_dec)
            self.y.extend(output_dec)
            self.states.extend([initial_states[i]]*num_actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        
        encoder_x = torch.tensor(self.states[idx],dtype=torch.long)
        decoder_x = torch.tensor(self.x[idx],dtype=torch.long)
        decoder_y = torch.tensor(self.y[idx],dtype=torch.long)
        
        return {
            "encoder_x": encoder_x,  
            "decoder_x": decoder_x,  
            "decoder_y": decoder_y,  
            "shape_encoder": encoder_x.shape[0],
            "shape_decoder": decoder_x.shape[0],
        }