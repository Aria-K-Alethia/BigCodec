import typing
from typing import List
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange

class MultiResolutionMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None]
    ):
        super().__init__()
        self.mel_transforms = nn.ModuleList([
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=window_length,
                hop_length=window_length // 4,
                n_mels=n_mel,
                power=1.0,
                center=True,
                norm='slaney',
                mel_scale='slaney',
            )
            for n_mel, window_length in zip(n_mels, window_lengths)
        ])
        self.n_mels = n_mels
        self.loss_fn = nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow
    
    def forward(self, x, y):
        loss = 0.0
        for mel_transform in self.mel_transforms:
            x_mel = mel_transform(x)
            y_mel = mel_transform(y)
            log_x_mel = x_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            loss += self.loss_fn(log_x_mel, log_y_mel)
        return loss
