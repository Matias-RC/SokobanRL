
from dreamers.fantasizer import Fantasizer
from dreamers.replayer import Replayer

class Dreamer:

    def __init__(self):

        self.fantasizer = Fantasizer(
            data_generator="backward_traversal",
            trainer="dpra"
        )

    def do(self, session, model):
        
        new_model = self.fantasizer.do(session, model)
        
        return new_model
