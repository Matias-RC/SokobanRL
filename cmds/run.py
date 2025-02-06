import sys
sys.path.append(".")

from models.dreamcoder.agent import Agent
from data.task import Task
from data.env_objects.sokoban_scenario import Scenario
from learning.curriculum import Curriculum
from models.dreamcoder.q_uniform import q_uniform
from managers.sokoban_manager import SokobanManager

NUM_TASKS = 20

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

scenarios = [ Scenario(width=6, height=6) for _ in range(NUM_TASKS)] #room
print(scenarios)
session_1 = [
    Task(
        initial_state= init_state,
    ) for init_state in scenarios
]

curriculum = Curriculum(
    sessions={
        "S1": session_1
    },
    strategy = "sorted"
)

m = SokobanManager()
a = Agent(
    actions=actions_for_sokoban,
    manager=m,
    q_net=q_uniform,
    batchSize=1,
    drawSize=1
)

for key_sessions, session in curriculum.sessions.items():
    a.wake(m,session) # solve all the tasks in the session
    a.sleep()       # use each solution from the last session to learn patterns trought two-phases: (1) Abstraction and (2) Dreaming
    # (1) Abstraction: Found factors (macro-actions) to decrease the solver's search-space
    # (2) Dreaming: TODO future

print("Factors:", a.current_factors)
