import torch
import torch.nn as nn

class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, rank):
        """
        Compute the pairwise loss.

        Parameters:
        output (Tensor): Model output for state
        rank (Tensor): Known tensor representing the rank of the state

        Returns:
        Tensor: Computed pairwise loss
        """
        # Compute pairwise loss output is a tensor of size (batch_size, 1)
        ij_output = output - output.squeeze().unsqueeze(0)

        # ground-truth pairwise label
        ij_distance = rank.unsqueeze(1) - rank.unsqueeze(0)
        ij_label = (ij_distance < 0).float()

        # Upper triangular mask (witout diagonal)
        tri_u_mask = (torch.triu(torch.ones_like(ij_label), diagonal=1) == 1)

        loss = self.cross_entropy(ij_output[tri_u_mask], ij_label[tri_u_mask])

        return loss  # Mean reduction for batch training
