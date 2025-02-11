import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, List
import random
import math
from managers.sokoban_manager import Node, SokobanManager

# ================================
# Mock Delta Scorer
# ================================
class DummyDeltaScorer:
    def m(self, state, action_sequence):
        return 1  # Dummy uniform score
    
    def q(self, node, legal_actions):
        return [1 / len(legal_actions)] * len(legal_actions)  # Equal probability

# ================================
# Test Setup
# ================================

grid = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 3, 0, 4, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 2, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1]
])

initial_state = (4, 1)
actions = [(-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1)]
final_node = Node(state=(1, 4))
current = final_node

for action in reversed(actions):
    current = Node(state=(current.state[0] - action[0], current.state[1] - action[1]), parent=current, action=action)
