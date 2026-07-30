from torch import nn


class CenterCrop1D(nn.Module):
    def __init__(self, crop_length: int):
        super().__init__()
        self.crop_length = crop_length

    def forward(self, x):
        if x.size(-1) <= self.crop_length:
            return x

        start = (x.size(-1) - self.crop_length) // 2
        return x[..., start:start + self.crop_length]