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


episodes = 8
lr_c = 1e-3

critic = source.MLP(in_dim=52,hid_dim=26,out_dim=1, num_hidden_layers=2)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_c)

Seeds = source.MakeSeedsList(episodes)
GridDims = source.MakeDimsList(2,7,episodes)
BoxNums = source.RandVariablelist(1, 2, episodes)
WallsNums = source.RandVariablelist(2, 9, episodes)

logics = master()

for idx, ep in enumerate(Seeds):
    grid = source.GenerateEmptyGrid(GridDims[idx][0], GridDims[idx][1])
    grid = source.FillWithGoalBoxes(grid, BoxNums[idx], ep)
    grid = source.FillWithWalls(grid, WallsNums[idx], ep)
    grid = source.PlacePlayer(grid, ep)
    print(grid)
    posBox = logics.PosOfBoxes(grid)
    posWalls = logics.PosOfWalls(grid)
    posGoals = logics.PosOfGoals(grid)
    posPlayer = logics.PosOfPlayer(grid)
    InverTuple, PosiblePosBox = logics.legalInverts(posPlayer, posBox, posWalls, posGoals)
    print(InverTuple[0])
    posPlayer = logics.FastInvert(posPlayer,InverTuple[0])
    print(logics.legalInverts(posPlayer, PosiblePosBox[0], posWalls, posGoals)[0])