import torch
import torch.nn as nn

class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()

    def forward(self, output_si, output_sj, ij_distance):
        """
        Compute the pairwise loss.

        Parameters:
        output_si (Tensor): Model output for state i
        output_sj (Tensor): Model output for state j
        ij_distance (Tensor): Known tensor representing the distance between i and j

        Returns:
        Tensor: Computed pairwise loss
        """
        loss = -torch.log(1 / (1 + torch.exp(-(output_si - output_sj)))) * ij_distance
        return loss.mean()  # Mean reduction for batch training
