import numpy as np
from collections import defaultdict
import random

from trainers.dpra import DPRA
from data_generators.backward_traversal import BackwardTraversal
from trainers.dpra import DPRA

from models.transformers.task.scoring_model import TransformerEncoderForScoring
from managers.inverse_manager import InversedSokobanManager
from managers.sokoban_manager import SokobanManager


class Fantasizer:
    def __init__(self, data_generator,trainer):
        
        if data_generator == "backward_traversal":
            self.data_generator = BackwardTraversal(session=None,
                                                    model=None,
                                                    solver=None,
                                                    inverseManager=InversedSokobanManager(),
                                                    manager=SokobanManager(),)

        if trainer == "dpra":
            self.trainer = DPRA()
                
    def do(self, session, model):
        
        dataset = self.data_generator.do(session, model) #pytorch dataset

        trained_model = self.trainer.do(dataset, model) #model trained by the DPRA/other algorithm

        return trained_model
    