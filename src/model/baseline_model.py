import torch
from torch import nn
from torch.nn import Sequential

class MFM(nn.Module):
    """
    Max-Feature-Map activation.
    """

    def forward(self, x):
        first, second = torch.chunk(x, chunks=2, dim=1)
        return torch.maximum(first, second)

class BaselineModel(nn.Module):
    """
    LCNN model for audio classification.
    """

    def __init__(self, n_class=2):
        super().__init__()

        self.net = Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            MFM(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(48),

            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(48),

            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(64),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Flatten(),
            
            nn.Linear(
                in_features=1184,
                out_features=160,
),
            MFM(),

            nn.BatchNorm1d(80),

            nn.Linear(
                in_features=80,
                out_features=n_class,
            ),
        )


    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        if data_object.ndim == 3:
            data_object = data_object.unsqueeze(1)

        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
