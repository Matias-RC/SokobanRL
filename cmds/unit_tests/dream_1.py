import sys

# Add the global path where your module is located
sys.path.append(".")

import random

from models.dreamcoder.agent import Agent
from data.task import Task
from models.q_models.delta_scorer import DeltaScorer

    
a = Agent(
    actions=[],
    manager=None,
    q_net = DeltaScorer(),
    batchSize=10,
    drawSize=1
)

session = [Task()]

a.dreamer.do(
            session=session,
            model=a.q_net
        )