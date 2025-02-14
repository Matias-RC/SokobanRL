import random
from data.task_solver_interaction import Task_Solver_Interaction
from managers.sokoban_manager import SokobanManager
from dcLogic import Solver
import numpy as np
import heapq

class aStarSearch:
    def __init__(self, library_actions, manager, deepRankModel, max_depth=100):
        self.manager = manager
        self.deepRankModel = deepRankModel
        self.library_actions = library_actions
        self.max_depth = max_depth
        self.Heap = []  # Priority queue using heapq
        self.visited = set()  # Set to store visited states
    
    def heuristic(self, node):
        return self.deepRankModel.forward(node.state)

    def push(self, node, g_cost):
        h = self.heuristic(node)
        entry = (g_cost + h, g_cost, node)  # Priority is f = g + h
        heapq.heappush(self.Heap, entry)

    def pop(self):
        return heapq.heappop(self.Heap)  # O(log n) time complexity
    
    def isEmpty(self):
        return len(self.Heap) == 0

    def do(self, task):
        node = self.manager.initializer(task.initial_state)
        self.push(node, g_cost=0)
        depth = self.max_depth
        count = 0

        while not self.isEmpty() and depth > 0:
            _, g_cost, node = self.pop()
            
            # Check if the goal state is reached
            if self.manager.isEndState(node):
                print(count)
                return node.trajectory()  # Return the solution path

            # Avoid revisiting states
            if node.state in self.visited:
                continue  
            self.visited.add(node.state)

            # Expand the current node
            for action in self.library_actions:
                condition, new_node = self.manager.legalUpdate(node, action)
                if condition and new_node.state not in self.visited:
                    count += 1
                    self.push(new_node, g_cost + 1)  # Increase path cost
            
            depth -= 1
        
        return None  # Return None if no solution is found within the depth limit
