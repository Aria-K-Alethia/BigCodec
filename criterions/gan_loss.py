import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def disc_loss(self, real, fake):
        real_loss = F.mse_loss(real, torch.ones_like(real))
        fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        return real_loss, fake_loss

    def gen_loss(self, fake):
        gen_loss = F.mse_loss(fake, torch.ones_like(fake))
        return gen_loss
