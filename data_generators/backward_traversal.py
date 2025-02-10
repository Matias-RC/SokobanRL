
class BackwardTraversal:
    def __init__(self, session, model, manager, maximumDepth, testsPerSearch, inverseManager):
        self.session  = session
        self.model =  model
        self.manager = manager
        self.maxDepth = maximumDepth
        self.cadence = testsPerSearch
        self.inverseManager = inverseManager
    def generate():
        batch = []
        
    def do(self, session, model):
        dataset = []
        for i in session:
            terminal, initialState = i
            self.manager.initializer(initialState)
            self.inverseManager.initializer(initialState, terminal)
            dataset.append(self.generate())
        return dataset