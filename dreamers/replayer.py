import math
import random 

class Replayer:

    def __init__(self, succesful_trajectories, agent):
        self.succesful_trajectories = succesful_trajectories
        self.agent = agent
    
    def pairwise_loss(self, i, ij_distance):
        """
        Compute the pairwise loss of the model with the succesful trajectories
        Theory:
        \mathcal{L}_{pair} = -[y\log(P_{i,j})+(1-y)\log(1-P_{i,j})]
        y  = (1 if $r(x_i) > r(x_j)$ and 0 otherwise)
        """
        return - math.log(i)*ij_distance # Entropy of i because  we assume it is the better state relative to j and that y = 1

    def do(self, num_comparasions):
        """
        We want to compute the loss of the model with deep pairwise rank aggregation (DPRA)
        ->For this we need to compute the pairwise loss of the model with the succesful trajectories
        """
        for trajectory in self.succesful_trajectories:
            #compute i = self.agent.m(state_i) for all i in trajectory
            generated_ranks = [self.agent.m(state) for state in trajectory]
            actual_ranks = reversed([i for i in range(trajectory)])
            for i in range(len(trajectory)):
                used_indices = set()
                for j in range(num_comparasions):
                    random_index = random.randint(j+1, len(trajectory)-1)
                    while random_index in used_indices:
                        random_index = random.randint(j+1, len(trajectory)-1)
                    ij_distance = actual_ranks[i] - actual_ranks[random_index]
                    loss = self.pairwise_loss(i, ij_distance)
        updated_params = self.agent.q_net.update(loss)

        return updated_params