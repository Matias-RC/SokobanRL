import numpy as np
from collections import defaultdict
import random

from trainers.dpra import DPRA
from data_generators.backward_traversal import BackwardTraversal
from trainers.dpra import DPRA

class Fantasizer:
    def __init__(self, data_generator, trainer):
        
        if data_generator == "backward_traversal":
            self.data_generator = BackwardTraversal()

        if trainer == "dpra":
            self.trainer = DPRA()
                
    def do(self, session, model):

        #dataset of pytorch
        dataset = self.data_generator.do(session, model)

        trained_model = self.trainer.do(dataset, model)

        return trained_model
    