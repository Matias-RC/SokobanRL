import torch
from torch.utils.data import Dataset
import numpy as np
import random

class GenerativeDataset(Dataset):

    '''GenerativeDataset recieves task and generate grid(x), actions(y) pairs'''
    
    def __init__(self,
                 session_batch,
                 samples_per_session=5,
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
            size_instance = len(actions_seq)
            
            for k in range(samples_per_session):

                input_dec, output_dec = actions_seq[k:(k+block_size)], actions_seq[(k+1):(k+block_size+1)]
                self.all_actions.append(actions_seq)
                self.y.append(input_dec)
                self.x.append(output_dec)
                self.states.append(initial_states[i])
        print(0/0) ##
    
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