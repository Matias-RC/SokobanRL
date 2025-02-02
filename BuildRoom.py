import numpy as np
import pickle
import SokoSource as source
from Logic import master

logic = master(source.heuristic,source.cost)

# Load templates and validate
with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)


height = 9
width = 9
# Print the generated level
inner_room = source.BuildRoom(width,height, templates)
room = np.ones((height + 2, width + 2), dtype=inner_room.dtype)
room[1:-1, 1:-1] = inner_room
room = source.FillWithGoalBoxes(room, 5, seed=None)
room = source.PlacePlayer(room)

print(room)

posPlayer = logic.PosOfPlayer(room)
posBox = logic.PosOfBoxes(room)
envPosGoals = logic.PosOfGoals(room)
envPosWalls = logic.PosOfWalls(room)

logic.posGoals = envPosGoals
logic.posWalls = envPosWalls

logic.DepthAndBreadthLimitedSearch(posPlayer, posBox, 3000, 10000)

cache_contents = logic.get_longest_solution_from_cache()
print(cache_contents)
newRoom = source.create_environment(room.shape,envPosWalls, envPosGoals, cache_contents[0])
print(newRoom)