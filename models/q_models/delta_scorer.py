import torch

from models.transformers.task.scoring_model import TransformerEncoderForScoring

class DeltaScorer:

    def __init__(self):
        
        self.scorer = TransformerEncoderForScoring()
        self.compiler = ...

    def q(self, example, program): # example is a tuple of (current_state, previous_actions), program is a list of actions
        
        output = self.compiler.do(example, program)
        
        return torch.sigmoid(self.scorer(output) - self.scorer(example))
    
    def get_learner(self):
        return self.scorer
    
    def set_learner(self, learner):
        self.scorer = learner
        return self