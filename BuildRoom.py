import numpy as np
import pickle

from regex import B
import BuildRoomSource as source

# Load templates and validate
with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)



# Print the generated level
print(source.BuildRoom(20, 13, templates))
