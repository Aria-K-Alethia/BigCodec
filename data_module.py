import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg)
        dl = DataLoader(ds, batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=8,
                        collate_fn=ds.collate_fn,
                        persistent_workers=True)

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

class FSDataset(Dataset):
    """FastSpeech dataset batching text, mel, pitch 
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()
        
        self.sr = cfg.preprocess.audio.sr
        
        self.filelist = utils.read_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.min_audio_length = cfg.dataset.min_audio_length
        
    def __len__(self):
        return len(self.filelist)

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav
    
    def __getitem__(self, idx):
        (fid, wavpath) = self.filelist[idx]
        wavpath = join(self.cfg.preprocess.datasets.LibriSpeech.root, wavpath)
        wav = self.load_wav(wavpath)
        wav = torch.from_numpy(wav)
        length = wav.shape[0]
        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[0]
        i = random.randint(0, length-self.min_audio_length)
        wav = wav[i:i+self.min_audio_length]

        out = {
            'fid': fid,
            'wav': wav
        }
        
        return out
    
    def collate_fn(self, bs):
        fids = [b['fid'] for b in bs]
        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)

        out = {
            'fid': fids,
            'wav': wavs
        }
        return out
        
        
