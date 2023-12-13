import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        squared_diff = (output - target) ** 2
        weighted_squared_diff = squared_diff * self.weights
        loss = weighted_squared_diff.mean()
        return loss

