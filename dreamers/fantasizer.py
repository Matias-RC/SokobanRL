import numpy as np
from collections import defaultdict
import random

from trainers.dpra import DPRA
from data_generators.backward_traversal import BackwardTraversal
from trainers.dpra import DPRA

from models.transformers.task.scoring_model import ScoringModel
from managers.inverse_manager import InversedSokobanManager
from managers.sokoban_manager import SokobanManager


class Fantasizer:
    def __init__(self, data_generator,trainer,model,solver):
        
        if data_generator == "backward_traversal":
            self.data_generator = BackwardTraversal(session=None,
                                                    model=model,
                                                    solver=solver,
                                                    inverseManager=InversedSokobanManager(),
                                                    manager=SokobanManager(),)

        if trainer == "dpra":
            self.trainer = DPRA()
                
    def do(self, session, model):

        #dataset of pytorch
        dataset = self.data_generator.do(session, model)

        trained_model = self.trainer.do(dataset, model)

        return trained_model
    