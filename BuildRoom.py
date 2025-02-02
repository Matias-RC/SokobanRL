import numpy as np
import pickle
import SokoSource as source

# Load templates and validate
with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)



# Print the generated level
room = source.BuildRoom(20, 13, templates)
room = source.FillWithGoalBoxes(room, 4, seed=None)
print(room)
