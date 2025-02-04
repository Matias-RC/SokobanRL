import sys

# Add the global path where your module is located
sys.path.append(".")

import random

from models.dreamcoder.agent import Agent
from data.task import Task

def generate_trajectories(num_trajectories, primitives, pattern, min_length=6, max_length=12):
    trajectories = []
    for _ in range(num_trajectories):
        length = random.randint(min_length, max_length - len(pattern)) + len(pattern)
        trajectory = []
        insert_index = random.randint(0, length - len(pattern))
        
        while len(trajectory) < length:
            if len(trajectory) == insert_index:
                trajectory.extend(pattern)
            else:
                trajectory.append(random.choice(primitives))
        
        trajectories.append(trajectory)
    
    return trajectories

primitives = ["up", "down", "left", "right"]
pattern = ["up", "left", "up", "right"]
num_trajectories = 6

trajectories = generate_trajectories(num_trajectories, primitives, pattern)
for t in trajectories:
    print(t)

session = []

for trajectory in trajectories:
    t = Task(
        scenario=None,
        objetive=None,
        initial_state=None
    )
    t.add(trajectory)
    session.append(t)
    
a = Agent()
a.current_session = session

a.abstraction()

print("Factors:", a.current_factors)

