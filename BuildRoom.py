import numpy as np
import pickle
import SokoSource as source

# Load templates and validate
with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)



# Print the generated level
room = source.BuildRoom(8, 6, templates)
room = np.pad(room, pad_width=1, mode='constant', constant_values=1)
room = source.FillWithGoalBoxes(room, 2, seed=None)
room = source.PlacePlayer(room)
print(room)