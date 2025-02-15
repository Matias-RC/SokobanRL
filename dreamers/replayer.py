import math
import random 
from trainers.dpra import DPRA


class Replayer:

    def __init__(self, succesful_trajectories, agent, method):
        self.succesful_trajectories = succesful_trajectories
        self.agent = agent
        if method == "pairwise_loss":
            self.train_with_trayectory = DPRA.trainWithTrajectory
        else:
            self.train_with_trayectory = DPRA.trainWithTrajectory


    
    def pairwise_loss(self, i,j, ij_distance):
        """
        Compute the pairwise loss of the model with the succesful trajectories
        Theory:
        \mathcal{L}_{pair} = -[y\log(P_{i,j})+(1-y)\log(1-P_{i,j})]
        y  = (1 if $r(x_i) > r(x_j)$ and 0 otherwise)
        """
        return - math.log(1/(1+math.exp(-(i-j))))*ij_distance # Entropy of i because  we assume it is the better state relative to j and that y = 1

    def do(self, usage_quota):
        for t in self.succesful_trajectories:
            loss = self.train_with_trayectory(t, usage_quota)
            
