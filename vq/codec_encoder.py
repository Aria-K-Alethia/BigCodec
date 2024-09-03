import torch
from torch import nn
import numpy as np
from .module import WNConv1d, EncoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 2, 5, 5),
                dilations=(1, 3, 9),
                out_channels=1024):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations)]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        # Create last convolution
        self.block += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)