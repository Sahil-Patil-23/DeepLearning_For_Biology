import torch.nn.functional as F
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard BCE loss
        # We use 'none' to keep the loss for each label separate before weighting
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get the probability of the true class
        p_t = torch.exp(-ce_loss) 
        
        # Calculate Focal Loss: (1-p_t)^gamma * BCE
        focal_loss = self.alpha * (1 - p_t)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss