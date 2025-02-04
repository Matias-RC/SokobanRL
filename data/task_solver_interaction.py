from data.task import Task


class Task_Solver_Interaction:
    def __init__(self,task):

        self.task = task
        self.position = task.initial_state
    
    def update_position(self,action):
        if action == "right":
            self.position = self.position[0] + 1, self.position[1]
        elif action == "up":
            self.position = self.position[0], self.position[1] + 1
        elif action == "left":
            self.position = self.position[0] - 1, self.position[1]
        elif action == "down":
            self.position = self.position[0] , self.position[1] - 1 
        
    def is_correct_final_state(self):
        return self.position == self.task.objective
    
    def is_fail_state(self):
        return self.position in self.task.failure_states
    
    
