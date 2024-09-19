import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.audio import stft
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d
from einops import rearrange

class SpecDiscriminator(nn.Module):
    def __init__(self,
                 stft_params=None,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=(7, 3),
                 channels=32,
                 max_downsample_channels=512,
                 downsample_scales=(2, 2, 2),
                 use_weight_norm=True,
                 ):
        super().__init__()

        if stft_params is None:
            stft_params = {
                'fft_sizes': [1024, 2048, 512],
                'hop_sizes': [120, 240, 50],
                'win_lengths': [600, 1200, 240],
                'window': 'hann_window'
            }

        self.stft_params = stft_params
        
        self.model = nn.ModuleDict()
        for i in range(len(stft_params['fft_sizes'])):
            self.model["disc_" + str(i)] = NLayerSpecDiscriminator(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                channels=channels,
                max_downsample_channels=max_downsample_channels,
                downsample_scales=downsample_scales,
            )

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        results = []
        i = 0
        x = x.squeeze(1)
        for _, disc in self.model.items():
            spec = stft(x, self.stft_params['fft_sizes'][i], self.stft_params['hop_sizes'][i],
                        self.stft_params['win_lengths'][i],
                        window=getattr(torch, self.stft_params['window'])(self.stft_params['win_lengths'][i])) # [B, T, F]
            spec = spec.transpose(1, 2).unsqueeze(1) # [B, 1, F, T]
            results.append(disc(spec))
            i += 1
        return results

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
        self.apply(_reset_parameters)


class NLayerSpecDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=(5, 3),
                 channels=32,
                 max_downsample_channels=512,
                 downsample_scales=(2, 2, 2)):
        super().__init__()

        # check kernel size is valid
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.Conv2d(in_channels, channels,
                      kernel_size=kernel_sizes[0],
                      stride=2,
                      padding=kernel_sizes[0] // 2),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = channels
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)

            model[f"layer_{i + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(downsample_scales) + 1}"] = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=kernel_sizes[1],
                      padding=kernel_sizes[1] // 2),
            nn.LeakyReLU(0.2, True),
        )

        model[f"layer_{len(downsample_scales) + 2}"] = nn.Conv2d(
            out_chs, out_channels, kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2)

        self.model = model

    def forward(self, x):
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results

