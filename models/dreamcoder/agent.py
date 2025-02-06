from solvers.tree import MonteCarloTreeSearch
from abstractors.bayesian import Decompiling
from simAnnaeling import kOpt
from data.task import Task

class Agent:
    def __init__(self,actions,manager,q_net,batchSize,drawSize):

        self.actions = actions

        self.current_session = None    
        self.current_factors = None
        
        self.q_net=q_net

        self.solver=MonteCarloTreeSearch(library_actions=self.actions,
                                         manager=manager,
                                         batchSize=batchSize,
                                         drawSize=drawSize)

        self.library = self.actions

        self.abstractor = Decompiling()
        
    
    def wake(self, wake_manager, session: list[Task]):

        self.current_session = session
        
        for task in session:
            
            solution = self.solve(task)

            task.add(solution)

            if solution is not None:
                print("Non optimized solution:")
                print(solution.trajectory())
                print("Optimized Solution:")
                print(kOpt(task.initial_state, solution, 3, self.actions, wake_manager))
            else:
                print("X")


    def sleep(self):

        self.abstraction()
        # TODO: self.dreaming()       

    def solve(self, task):

        return self.solver.do(task,self.q_net)

    def abstraction(self):

        self.current_factors = self.abstractor.do(
            session=self.current_session,
            k=2,
            vocabulary_size=len(self.library)
        )

        self.refact()
        
    def refact(self):

        # self.library += factors # Expand using abstractions/factors

        pass
    