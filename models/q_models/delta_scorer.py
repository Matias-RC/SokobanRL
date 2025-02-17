import torch

from models.transformers.task.scoring_model import TransformerEncoderForScoring

class DeltaScorer:

    def __init__(self,library, scorer="transformer", config=None, compiler=None):
        self.library = library
        if scorer == "transformer":
            self.scorer = TransformerEncoderForScoring(hidden_dim=64,
                                                       num_layers=2,
                                                       num_heads=1,
                                                       embedding_norm_scalar=1,
                                                       use_norm=False,
                                                       use_attention_dropout=True,
                                                       eps=0.000001,
                                                       share_layers=False,
                                                       device="cpu",
                                                       embedding_type="learnable",
                                                       attention_type="standard",
                                                       output_dim=1)
        
        self.compiler = compiler

    def q(self, example, program): # example is a tuple of (current_state, previous_actions), program is a list of actions
        
        output = self.compiler.do(example, program)
        
        return torch.sigmoid(self.scorer(output) - self.scorer(example))
    
    def get_learner(self):
        return self.scorer
    
    def set_learner(self, learner):
        self.scorer = learner
        return self