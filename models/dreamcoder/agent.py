from solvers.tree import MonteCarloTreeSearch
from abstractors.bayesian import Decompiling
#from simAnnaeling import kOpt
from data.task import Task
from dreamers.dreamer import Dreamer

class Agent:
    def __init__(self,actions,manager,model,recognition_model,batchSize,drawSize):

        self.actions = actions

        self.current_session = None    
        self.current_factors = None
        
        self.model = model 
        self.recognition_model = recognition_model 

        self.solver=MonteCarloTreeSearch(library_actions=self.actions,
                                         manager=manager,
                                         batchSize=batchSize,
                                         drawSize=drawSize)

        self.library = self.actions

        self.abstractor = Decompiling()

        self.dreamer = Dreamer(model)
        
    
    def wake(self, wake_manager, session: list[Task]):

        self.current_session = session
        
        for task in session:
            
            solution = self.solve(task) #solution is a node

            task.add(solution)

            if solution is not None:
                print(solution.trajectory())
            else:
                print("X")
        
        return session


    def sleep(self):
        self.dreaming()
        self.abstraction()
        # self.dreaming()       

    def solve(self, task):

        return self.solver.do(task,self.model) #return a solution

    def abstraction(self):

        self.current_factors = self.abstractor.do(
            session=self.current_session,
            k=2,
            vocabulary_size=len(self.library)
        )

        self.refact()
        
    def refact(self,factors=[]):

        self.library += factors # Expand using abstractions/factors
    
    def dreaming(self):
        
        self.model = self.dreamer.do(
            session=self.current_session,
            model=self.recognition_model
        )