import sys
sys.path.append(".")

from models.dreamcoder.agent import Agent
from data.task import Task
from data.env_objects.sokoban_scenario import Scenario
from learning.curriculum import Curriculum
from models.dreamcoder.q_uniform import q_uniform
from managers.sokoban_manager import SokobanManager
from models.transformers.task.scoring_model import TransformerEncoderForScoring

NUM_TASKS = 2
GRID_SIZE = 6

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

scenarios = [ Scenario(width=GRID_SIZE, height=GRID_SIZE) for _ in range(NUM_TASKS)] #room

#print("Scenarios:", scenarios)

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


model = TransformerEncoderForScoring(hidden_dim=64,
                                    num_layers=1,
                                    num_heads=1,
                                    embedding_norm_scalar=1,
                                    use_norm=False,
                                    use_attention_dropout=True,
                                    eps=0.000001,
                                    share_layers=False,
                                    device="cpu",
                                    embedding_type="learnable",
                                    attention_type="standard",
                                    output_dim=1)


m = SokobanManager()
a = Agent(
    actions=actions_for_sokoban,
    manager=m,
    model=model,
    batchSize=10,
    drawSize=1
)

for key_sessions, session in curriculum.sessions.items():
    session_solved = a.wake(m,session) # solve all the tasks in the session
    print(session_solved[0])
    a.sleep()       # use each solution from the last session to learn patterns trought two-phases: (1) Abstraction and (2) Dreaming
    # (1) Abstraction: Found factors (macro-actions) to decrease the solver's search-space
    # (2) Dreaming: TODO future

print("Factors:", a.current_factors)
