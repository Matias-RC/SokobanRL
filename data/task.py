
class Task:
    def __init__(self,initial_state):

        self.initial_state = initial_state
        self.is_solved = False

    def add(self, solution):
        if solution is not None:
            self.solution = solution
            self.is_solved = True

    