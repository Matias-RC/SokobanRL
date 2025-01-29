import math

# Arr : Arrengement of goals and boxes
# Batch : Set of arrengements

def NumberOfBatches(grid, c):
    y, x = grid
    return int(math.sqrt(y*x) + c)

