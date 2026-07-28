import random

from torch import nn

class RandomCrop1D(nn.Module):
    """
    Randomly crop a 1D tensor to a specified length.

    Args:
        crop_length (int): The length of the cropped output tensor.
    """

    def __init__(self, crop_length: int):
        super().__init__()
        self.crop_length = crop_length

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor of shape [channels, samples].

        Returns:
            x (Tensor): cropped tensor.
        """
        if x.size(-1) <= self.crop_length:
            return x

        start = random.randint(0, x.size(-1) - self.crop_length)
        return x[..., start:start + self.crop_length]
    