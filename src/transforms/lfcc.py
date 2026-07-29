import torchaudio
import torch
from torch import nn

class LFCC(nn.Module):
    """
    Convert waveform into LFCC
    """

    def __init__(self, sample_rate = 16000, n_lfcc = 20, n_filter = 20, n_fft = 512, win_length = 320, hop_length = 160):
        super().__init__()
        self.transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            log_lf=True,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "power": 2.0,
                "normalized": False,
            },
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): waveform[channels, samples].

        Returns:
            Tensor: LFCC [channels, n_lfcc, time].
        """
        x = self.transform(x)
        return x