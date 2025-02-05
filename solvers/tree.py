import random
from data.task_solver_interaction import Task_Solver_Interaction
from managers.sokoban_manager import SokobanManager
from dcLogic import Solver #solver for sokoban

class MonteCarloTreeSearch:
    def __init__(self, library_actions):
        self.manager = None
        self.library_actions = library_actions

    def tree_search(self, task, manager_dictionary,  ):
        self.manager = manager_dictionary[task.key]
        game_data = task.initial_state,
        
        for action in self.library_actions:
            bool_condition, game_data = self.manager.LegalUpdate(macro=action,game_data=game_data)
    