import sys
sys.path.append(".")

from models.dreamcoder.agent import Agent
from data.task import Task
from data.env_objects.sokoban_scenario import Scenario
from learning.curriculum import Curriculum
from models.dreamcoder.q_uniform import q_uniform
from managers.sokoban_manager import SokobanManager

# deep q network
from DeepQNetwork.models.mlp import DQN

deepQNetwork = DQN()

#scenarios and tasks
NUM_TASKS = 20

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

scenarios = [ Scenario(width=8, height=8) for _ in range(NUM_TASKS)] #room

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

#sokoban manager and agent
m = SokobanManager()

a = Agent(
    actions=actions_for_sokoban,
    manager=m,
    q_net= deepQNetwork,
    batchSize=10,
    drawSize=1
)


