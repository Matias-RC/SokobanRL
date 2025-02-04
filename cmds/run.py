from models.dreamcoder.agent import Agent
from data.task import Task
from data.env_objects.sokoban_scenario import Scenario
from learning.curriculum import Curriculum


scenarios = [ Scenario() for _ in range(1)]

session_1 = [
    Task(
        scenario= s,
        objective= s.get_objetive() ,
        initial_state= s.get_pos_player() ,
    ) for s in scenarios
]

curriculum = Curriculum(
    sessions={
        "S1": session_1
    },
    strategy = "sorted"
)

a = Agent()

for session in curriculum:
    a.wake(session) # solve all the tasks in the session
    a.sleep()       # use each solution from the last session to learn patterns trought two-phases: (1) Abstraction and (2) Dreaming
    # (1) Abstraction: Found factors (macro-actions) to decrease the solver's search-space
    # (2) Dreaming: TODO future