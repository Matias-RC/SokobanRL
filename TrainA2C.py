import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from Logic import master
import SokoSource as source

"""
EzEps = {ep1,...,epn}
ep_i = easy grid
-> since we need a lot of easy examples we store the seeds for repetability

for i in batches:
-for j in Eazy Episodes:
-- Stage the A2C is put up with depends of i
    - if i = 0 maybe only at the begining 
    - elif i > 1 can also be at a start but allow for deeper game states
    - Never making the BFS longer than 1000
-- make triple depth inversing with the State presented to the A2C
-- BackProp loss
-for j in Mid/Hard Episodes:
-(This is the same as before but the depth is only 2 and works with Astar)
"""


