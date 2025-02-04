import random
from data.task_solver_interaction import Task_Solver_Interaction


class MonteCarloTreeSearch:
    def __init__(self, actions):
        self.actions = actions

    def do(self, task):
        done = False
        solution = None
        task_solver = Task_Solver_Interaction(task)
        
        while not done:
            possible_rnd_solution = [] #possible solution
            while not task_solver.is_correct_final_state() or not task_solver.is_fail_state():
                next_action = random.choice(self.actions)
                task_solver.update_position(next_action)  #update state
                possible_rnd_solution.append(next_action)
            
            if task_solver.is_correct_final_state():
                solution = possible_rnd_solution
                done = True
        
        return solution 