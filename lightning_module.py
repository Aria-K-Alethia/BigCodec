import os
import random
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from vq import CodecEncoder, CodecDecoder
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss
from common.schedulers import WarmupLR

class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ocwd = hydra.utils.get_original_cwd()
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def construct_model(self):
        enccfg = self.cfg.model.codec_encoder
        encoder = CodecEncoder(
                    ngf=enccfg.ngf,
                    use_rnn=enccfg.use_rnn,
                    rnn_bidirectional=enccfg.rnn_bidirectional,
                    rnn_num_layers=enccfg.rnn_num_layers,
                    up_ratios=enccfg.up_ratios,
                    dilations=enccfg.dilations,
                    out_channels=enccfg.out_channels
                )
        deccfg = self.cfg.model.codec_decoder
        decoder = CodecDecoder(
                    in_channels=deccfg.in_channels,
                    upsample_initial_channel=deccfg.upsample_initial_channel,
                    ngf=deccfg.ngf,
                    use_rnn=deccfg.use_rnn,
                    rnn_bidirectional=deccfg.rnn_bidirectional,
                    rnn_num_layers=deccfg.rnn_num_layers,
                    up_ratios=deccfg.up_ratios,
                    dilations=deccfg.dilations,
                    vq_num_quantizers=deccfg.vq_num_quantizers,
                    vq_dim=deccfg.vq_dim,
                    vq_commit_weight=deccfg.vq_commit_weight,
                    vq_full_commit_loss=deccfg.vq_full_commit_loss,
                    codebook_size=deccfg.codebook_size,
                    codebook_dim=deccfg.codebook_dim,
                )
        mpdcfg = self.cfg.model.mpd
        mpd = HiFiGANMultiPeriodDiscriminator(
                    periods=mpdcfg.periods,
                    max_downsample_channels=mpdcfg.max_downsample_channels,
                    channels=mpdcfg.channels,
                    channel_increasing_factor=mpdcfg.channel_increasing_factor,
                )
        mstftcfg = self.cfg.model.mstft
        mstft = SpecDiscriminator(
                    stft_params=mstftcfg.stft_params,
                    in_channels=mstftcfg.in_channels,
                    out_channels=mstftcfg.out_channels,
                    kernel_sizes=mstftcfg.kernel_sizes,
                    channels=mstftcfg.channels,
                    max_downsample_channels=mstftcfg.max_downsample_channels,
                    downsample_scales=mstftcfg.downsample_scales,
                    use_weight_norm=mstftcfg.use_weight_norm,
                )
        model = nn.ModuleDict({
                    'CodecEnc': encoder,
                    'generator': decoder,
                    'discriminator': mpd,
                    'spec_discriminator': mstft,
                })
        print(model)
        self.model = model

    def construct_criteria(self):
        cfg = self.cfg.train
        criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.preprocess.audio.sr)
        if cfg.use_feat_match_loss:
            criteria['fm_loss'] = nn.L1Loss()
        criteria['gan_loss'] = GANLoss()
        criteria['l1_loss'] = torch.nn.L1Loss()
        criteria['l2_loss'] = torch.nn.MSELoss()
        self.criteria = criteria
        print(criteria)

    def forward(self, batch):
        wav = batch['wav']
        vq_emb = self.model['CodecEnc'](wav.unsqueeze(1))
        vq_post_emb, vq_code, vq_loss = self.model['generator'](vq_emb, vq=True)
        y_ = self.model['generator'](vq_post_emb, vq=False) # [B, 1, T]
        y = wav.unsqueeze(1)
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code
        }
        return output
    
    @torch.inference_mode()
    def inference(self, wav):
        vq_emb = self.model['CodecEnc'](wav.unsqueeze(1))
        vq_post_emb, vq_code, vq_loss = self.model['generator'](vq_emb, vq=True)
        y_ = self.model['generator'](vq_post_emb, vq=False).squeeze(1) # [B, T]
        return y_

    def compute_disc_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        p = self.model['discriminator'](y)
        p_ = self.model['discriminator'](y_)
        
        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if 'spec_discriminator' in self.model:
            sd_p = self.model['spec_discriminator'](y)
            sd_p_ = self.model['spec_discriminator'](y_)

            for i in range(len(sd_p)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(sd_p[i][-1], sd_p_[i][-1])
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)
        
        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss
        
        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }
        return output
    
    def compute_gen_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        vq_loss, vq_code = output['vq_loss'], output['vq_code']
        gen_loss = .0
        self.set_discriminator_gradients(False)
        output = {}
        cfg = self.cfg.train
        
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output['mel_loss'] = mel_loss
        
        # gan loss
        p_ = self.model['discriminator'](y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
        if 'spec_discriminator' in self.model:
            sd_p_ = self.model['spec_discriminator'](y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(sd_p_[i][-1]))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        output['adv_loss'] = adv_loss

        # fm loss
        if cfg.use_feat_match_loss:
            fm_loss = 0.
            with torch.no_grad():
                p = self.model['discriminator'](y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            output['fm_loss'] = fm_loss
            if 'spec_discriminator' in self.model:
                spec_fm_loss = 0.
                with torch.no_grad():
                    sd_p = self.model['spec_discriminator'](y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output['spec_fm_loss'] = spec_fm_loss

        # vq
        if vq_loss is not None:
            vq_loss = sum(vq_loss)
            gen_loss += vq_loss
            output['vq_loss'] = vq_loss
        
        self.set_discriminator_gradients(True)
        output['gen_loss'] = gen_loss
        return output
    
    def training_step(self, batch, batch_idx):
        output = self(batch)
        
        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()
        
        # discriminator 
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss']
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        self.clip_gradients(disc_opt, gradient_clip_val=self.cfg.train.disc_grad_clip, gradient_clip_algorithm='norm')
        disc_opt.step()
        disc_sche.step()

        # generator
        gen_losses = self.compute_gen_loss(batch, output)
        gen_loss = gen_losses['gen_loss']
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
        gen_opt.step()
        gen_sche.step()

        self.log_dict(disc_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)
        self.log_dict(gen_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)    
    
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
        
    def configure_optimizers(self):
        from itertools import chain
        disc_params = self.model['discriminator'].parameters()
        if 'spec_discriminator' in self.model:
            disc_params = chain(disc_params, self.model['spec_discriminator'].parameters())
        gen_params = chain(self.model['CodecEnc'].parameters(), self.model['generator'].parameters())
        
        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        print(f'Generator optim: {gen_opt}')
        print(f'Discriminator optim: {disc_opt}')
        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.model['discriminator'].parameters():
            p.requires_grad = flag
        
        if 'spec_discriminator' in self.model:
            for p in self.model['spec_discriminator'].parameters():
                p.requires_grad = flag
