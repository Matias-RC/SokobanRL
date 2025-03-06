# Here we test the generative model using a transformer encoder-decoder model
import sys
sys.path.append(".")

from models.dreamcoder.agent import Agent
from data.task import Task
from data.env_objects.sokoban_scenario import Scenario
from learning.curriculum import Curriculum
from models.dreamcoder.q_uniform import q_uniform
from managers.sokoban_manager import SokobanManager
from models.q_models.delta_scorer import DeltaScorer
import numpy as np

NUM_TASKS = 2
GRID_SIZE = 6

actions_for_sokoban = [
    [(-1, 0)],  # 'w' (UP)
    [(1, 0)],   # 's' (DOWN)
    [(0, -1)],  # 'a' (LEFT)
    [(0, 1)]    # 'd' (RIGHT)
]

scenarios = [ Scenario(width=GRID_SIZE, height=GRID_SIZE) for _ in range(NUM_TASKS)] #room

session_1 = [
    Task(
        initial_state= init_state,
    ) for init_state in scenarios
]

curriculum = Curriculum(
    sessions={
        "S1": session_1
    },
    strategy = "sorted"
)

#model = DeltaScorer(actions_for_sokoban)
m = SokobanManager()
a = Agent(
    actions=actions_for_sokoban,
    manager=m,
    model=None,
    recognition_model=None,
    batchSize=10,
    drawSize=1
)

for key_sessions, session in curriculum.sessions.items():
    session_solved = a.wake(m,session) # solve all the tasks in the session
    #print(session_solved[0])
    

''' Testing just one example on encoder-decoder transformer with standard attention '''
from models.transformers.task.generative_model import GenerativeModelTransformer
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 10
eval_interval = 2
learning_rate = 3e-4
device ="cuda:0" #'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 2
dropout = 0.1

#model with fix hyperparams
generative_model = GenerativeModelTransformer(hidden_dim = n_embd,
                                              num_layers = n_layer,
                                              num_heads = n_head,
                                              dropout_rate= dropout,
                                              embedding_norm_scalar= 1.0,
                                              use_norm = True,
                                              use_attention_dropout = False,
                                              eps = 1e-6,
                                              share_layers = False,
                                              device = "cpu",
                                              embedding_type = "learnable",
                                              attention_type = "standard",
                                              library_dim = 4, # up, left, right, down
                                              block_size=block_size,
                                              
                                              )

#input-ouput pairs example
from data.datasets.generative_model.dataset import GenerativeDataset
from data.datasets.generative_model.collate import collate_fn
from torch.utils.data import DataLoader

session_test = session_solved[0]
#print("solution example:");print(f"class of solution {type(session_test.solution)}")
#print(f"dictionary of solution {session_test.solution.__dict__.keys()}")

batch = [session_test]
dataset = GenerativeDataset(session_batch=batch)
dataloader = DataLoader(dataset,shuffle=False,collate_fn=collate_fn,batch_size=1)

for k, example in enumerate(dataloader):
    #print("example in dataset: ")
    #for d, data in enumerate(dataset):
    #    if d == k:
    #        print(data)
    #print("-----------------/////////////")
    print("example in dataloader")
    print(example.keys())
    break
    
#    for k,v in example.items():
#        print("-----------------")
        
        #print(v.shape)
        #print(k, v)
        #print("-----------------")
#    break


#training example
print("training the model")
import torch
optimizer=torch.optim.AdamW(generative_model.parameters(), lr=learning_rate)
max_iters=10
criterion=torch.nn.CrossEntropyLoss()

for iter in range(max_iters):
    for batch in dataloader:
        logits=generative_model(batch)
        loss=criterion(logits,batch["decoder_target_ids"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



