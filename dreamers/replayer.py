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

    def do(self, usage_quota):
        for t in self.succesful_trajectories:
            loss = self.train_with_trayectory(t, usage_quota)

