import os
import pytorch_lightning as pl
import hydra
import librosa
import soundfile as sf
import torch
import numpy as np
from os.path import join, exists, dirname, basename
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from data_module import DataModule
from lightning_module import CodecLightningModule
from tqdm import tqdm
from glob import glob
from time import time

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='default', version_base=None)
def train(cfg):
    datamodule = DataModule(cfg)
    lm = CodecLightningModule.load_from_checkpoint(cfg.ckpt, cfg=cfg)
    lm.eval()
    sr = cfg.preprocess.audio.sr
    ckpt_dir = dirname(cfg.ckpt)
    device = torch.device('cuda')
    lm = lm.to(device)
        
    wav_paths = glob(join(cfg.input_dir, '*.wav'))
    print(f'Found {len(wav_paths)} wavs')
    target_wav_dir = cfg.output_dir
    os.makedirs(target_wav_dir, exist_ok=True)
    
    st = time()
    for wav_path in tqdm(wav_paths):
        target_wav_path = join(target_wav_dir, basename(wav_path))
        wav = librosa.load(wav_path, sr=sr)[0] 
        wav = torch.from_numpy(wav).unsqueeze(0).to(device)
        wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
        with torch.no_grad():
            vq_emb = lm.model['CodecEnc'](wav.unsqueeze(1))
            vq_post_emb, vq_code, _ = lm.model['generator'](vq_emb, vq=True)
            recon  = lm.model['generator'](vq_post_emb, vq=False).squeeze(1) # [B, T]
        recon = recon.squeeze(0).detach().cpu().numpy()
        sf.write(target_wav_path, recon, sr)
    print(f"Done, output dir: {target_wav_dir}")
    et = time()
    print(f'Inference ends, time: {(et-st)/60:.2f} mins')
    
if __name__ == '__main__':
    train()
