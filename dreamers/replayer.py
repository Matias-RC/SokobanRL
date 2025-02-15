import math
import random 
from trainers.dpra import DPRA


class Replayer:

    def __init__(self, agent, method):

        self.agent = agent
        if method == "pairwise_loss":
            self.train_with_trayectory = DPRA.trainWithTrajectory
        else:
            self.train_with_trayectory = DPRA.trainWithTrajectory

    def do(self, succesful_trajectories):
        for t in succesful_trajectories:
            loss = self.train_with_trayectory(t)

