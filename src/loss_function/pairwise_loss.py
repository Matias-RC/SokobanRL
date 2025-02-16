import torch
import torch.nn.functional as F

def pairwise_loss(output_si, output_sj, ij_distance):
    """
    Compute the pairwise loss using PyTorch.

    Parameters:
    output_si (Tensor): Model output for state i
    output_sj (Tensor): Model output for state j
    ij_distance (Tensor): Known tensor representing the distance between i and j

    Returns:
    Tensor: Computed pairwise loss
    """
    loss = - torch.log(1/(1+torch.exp(-(output_si-output_sj))))*ij_distance
    return loss.mean()  # Mean reduction

"""
    def pairwise_loss(self, i,j, ij_distance):
        Compute the pairwise loss of the model with the succesful trajectories
        Theory:
        P_{i,j} =  1/(1+exp(-(M(s_i)-M(s_j))))
        \mathcal{L}_{pair} = -[y\log(P_{i,j})+(1-y)\log(1-P_{i,j})]
        y  = (1 if $r(x_i) > r(x_j)$ and 0 otherwise)
        return - math.log(1/(1+math.exp(-(i-j))))*ij_distance # Entropy of i because  we assume it is the better state relative to j and that y = 1
    class pairwise_loss(nn.Module):
        def __init__(self):
            pass
        def pairwise_loss(self,i,j,distance):
            
"""