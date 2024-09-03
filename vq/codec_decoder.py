import numpy as np
import torch
import torch.nn as nn
from .residual_vq import ResidualVQ
from .module import WNConv1d, DecoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 upsample_initial_channel=1536,
                 ngf=48,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 up_ratios=(5, 5, 2, 2, 2),
                 dilations=(1, 3, 9),
                 vq_num_quantizers=1,
                 vq_dim=1024,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=8192,
                 codebook_dim=8,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        
        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )
        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        
        if use_rnn:
            layers += [
                ResLSTM(channels,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        
        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations)]

        layers += [
            Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        
        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.model(x)
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None
    
    def inference(self, x):
        x = self.model(x)
        return x, None


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
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
