#.venv imports
import numpy as np
#Local imports
import BuildRoomSource as source

templates_full = np.load("templates.npy")
corner_templates = np.load("corner_templates.npy")

templates = [corner_templates, templates_full]

def buildRoom(width,height):
    gridMatrix  = source.CreateEmpty(width, height)
    grid = source.PartitionGrid3x3(gridMatrix, templates)
