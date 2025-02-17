import sys
sys.path.append(".")

import numpy as np

from trainers.dpra import DPRA
from data.datasets.backward_traversal.dataset import BackwardTraversalDataset
from models.q_models.delta_scorer import DeltaScorer

grid_w = np.matrix([
    [1,1,1,1,1,1,1,1],
    [1,1,4,0,0,0,1,1],
    [1,1,4,0,3,3,2,1],
    [1,4,0,0,0,5,0,1],
    [1,0,0,3,1,1,0,1],
    [1,0,0,0,1,1,0,1],
    [1,1,1,0,3,4,0,1],
    [1,1,1,1,1,1,1,1]
])

rank_w = 10  

grid_b = np.matrix([
    [1,1,1,1,1,1,1,1],
    [1,1,4,0,3,2,1,1],
    [1,1,5,0,0,0,0,1],
    [1,5,0,0,0,4,0,1],
    [1,0,0,3,1,1,0,1],
    [1,0,0,0,1,1,0,1],
    [1,1,1,0,3,4,0,1],
    [1,1,1,1,1,1,1,1]
])

rank_b = 3

batch = [
    {
        "grid": grid_w,
        "rank": rank_w
    },
    {
        "grid": grid_b,
        "rank": rank_b
    }
]

trainer = DPRA()

datasets = []

batch_dataset_torch = BackwardTraversalDataset(batch)
datasets.append(batch_dataset_torch)

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]


model = DeltaScorer(actions_for_sokoban)


trainer.do(datasets, model) #model trained by the DPRA/other algorithm

