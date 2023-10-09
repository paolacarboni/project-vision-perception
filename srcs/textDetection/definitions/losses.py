import torch.nn as nn
import torch.nn.functional as F

class TextLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, mask):
        loss = F.binary_cross_entropy(prediction, mask.float())
        return loss
