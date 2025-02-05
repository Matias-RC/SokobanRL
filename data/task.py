
class Task:
    def __init__(self,scenario,key,initial_state):

        self.scenario = scenario
        self.key = key #ask to dictionary for update method
        self.initial_state = initial_state

    def add(self, solution):
        self.solution = solution

    