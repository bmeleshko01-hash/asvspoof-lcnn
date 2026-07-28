import math

from torch import nn

class RepeatPad1D(nn.Module):
    """
    Repeat the signal until it reaches the target length.
    """
    def __init__(self, target_length):
        self.target_length = target_length
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor of shape [channels, samples].

        Returns:
            x (Tensor): exit tensor.
        """
        if x.size(-1) >= self.target_length:
            return x
        repeats = math.ceil(self.target_length / x.size(-1))
        x = x.repeat(1, repeats)
        return x[..., : self.target_length]
