
from dreamers.fantasizer import Fantasizer
from dreamers.replayer import Replayer

class Dreamer:
    def __init__(self, agent):
        self.agent  = agent
        self.replayer = Replayer(
            agent=agent,
            method="pairwise_loss"
        )

        self.fantasizer = Fantasizer(
            data_generator="backward_traversal",
            trainer="dpra",
            agent=self.agent
        )

    def do(self, session, model):
        
        new_model = model #self.replayer.do(session, model)

        new_model = self.fantasizer.do(session, new_model)
        
        return new_model
