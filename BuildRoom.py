import numpy as np
import pickle
import BuildRoomSource as source

# Load templates and validate
with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)



# Print the generated level
print(source.BuildRoom(20, 13, templates))
