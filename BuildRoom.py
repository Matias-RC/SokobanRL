import numpy as np
import pickle

from regex import B
import BuildRoomSource as source

# Load templates and validate
with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)

def BuildRoom(level_width, level_height):
    level = source.construct_grid(level_width, level_height, templates)
    if not source.is_connected(level):
        return BuildRoom(level_width, level_height)
    return level

# Print the generated level
print(BuildRoom(9, 9))
