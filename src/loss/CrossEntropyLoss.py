import torch
from torch import device, nn


class CrossEntropyLoss(nn.Module):
    """
    Wrapper over PyTorch CrossEntropyLoss
    """

    def __init__(self):
        super().__init__()

        weights = torch.tensor([8.84, 1.0])
        self.loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits, labels)}
