import math
import torch
from torch import nn
from .factorized_vector_quantize import FactorizedVectorQuantize

class ResidualVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        codebook_size,
        **kwargs
    ):
        super().__init__()
        VQ = FactorizedVectorQuantize
        if type(codebook_size) == int:
            codebook_size = [codebook_size] * num_quantizers
        self.layers = nn.ModuleList([VQ(codebook_size=size, **kwargs) for size in codebook_size])
        self.num_quantizers = num_quantizers

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        
        for idx, layer in enumerate(self.layers):
            quantized, indices, loss = layer(residual)
            
            residual = residual - quantized
            
            quantized_out = quantized_out + quantized

            loss = loss.mean()

            all_indices.append(indices)
            all_losses.append(loss)
        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

    def vq2emb(self, vq, proj=True):
        # [B, T, num_quantizers]
        quantized_out = 0.
        for idx, layer in enumerate(self.layers):
            quantized = layer.vq2emb(vq[:, :, idx], proj=proj)
            quantized_out = quantized_out + quantized
        return quantized_out
    def get_emb(self):
        embs = [] 
        for idx, layer in enumerate(self.layers):
            embs.append(layer.get_emb())
        return embs
