
class Task:
    def __init__(self,scenario,objetive,initial_state):

        self.scenario = scenario
        self.objetive = objetive
        self.initial_state = initial_state

    def add(self, solution):
        self.solution = solution

    