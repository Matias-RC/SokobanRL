import random
from data.task_solver_interaction import Task_Solver_Interaction
from managers.sokoban_manager import SokobanManager
from dcLogic import Solver #solver for sokoban
import numpy as np


class aStarSearch:
    def __init__(self,library_actions, manager, deepRankModel,max_depth=100):
        self.manager = manager
        self.deepRankModel = deepRankModel
        self.library_actions = library_actions
        self.max_depth = max_depth
        self.Heap = []
        self.Count = 0
    
    def heuristic(self, node):
        return self.deepRankModel.foward(node.state)

    def push(self, node):
        self.Count += 1
        self.Heap.append(node)
        self.Heap.sort(key=lambda x: x[0])

    def pop(self):
        self.Count -= 1
        return self.Heap.pop(0)

    def isEmpty(self):
        return self.Count == 0

    def do(self,task):
        node = self.manager.initializer(task.initial_state)
        self.push((self.heuristic(node), node))
        depth = self.max_depth

        while not self.isEmpty() and depth > 0:
            _, node = self.pop()
            if self.manager.isEndState(node):
                return node.trajectory()
            for action in self.library_actions:
                condition, new_node = self.manager.legalUpdate(node, action)
                if condition:
                    self.push((self.heuristic(new_node), new_node))
            depth -= 1
        return None