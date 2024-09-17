import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        cross_entropy_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy_loss

        return focal_loss.mean()


class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, labels.float())

        return (loss, outputs) if return_outputs else loss
