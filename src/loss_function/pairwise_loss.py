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
