from data.task import Task

class Decompiling:

    def __init__(self, is_uniform=True):
        self.is_uniform = is_uniform

        if self.is_uniform:
            self.do = self.do_uniform

    def do_uniform(self, session: list[Task]):


        task = session[0]
        solution = task.solution

        factors = [] # Decompile the session to obtain factors
        
        return factors
    
    