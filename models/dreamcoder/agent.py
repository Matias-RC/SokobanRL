from solvers.tree import MonteCarloTreeSearch
from abstractors.bayesian import Decompiling

class Agent:
    def __init__(self):

        self.actions = {"left":3,"right":2,"up":1,"down":0}

        self.current_session = None    
        self.current_factors = None

        self.solver = MonteCarloTreeSearch(actions=self.actions)

        self.library = set(self.actions.keys())

        self.abstractor = Decompiling()
        
    
    def wake(self, session):

        self.current_session = session
        
        for task in session:
            
            solution = self.solve(task)

            task.add(solution)


    def sleep(self):

        self.abstraction()
        # TODO: self.dreaming()       

    def solve(self, task):

        return self.solver.do(task)

    def abstraction(self):

        self.current_factors = self.abstractor.do(
            session=self.current_session,
            k=1,
            vocabulary_size=len(self.library)
        )

        self.refact()
        
    def refact(self):

        # self.library += factors # Expand using abstractions/factors

        pass
    