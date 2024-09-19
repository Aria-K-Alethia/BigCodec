import os
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join, exists
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder import CodecDecoder
from argparse import ArgumentParser
from time import time

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='.')
    parser.add_argument('--ckpt', type=str, default='bigcodec.pt')
    parser.add_argument('--output-dir', required=True, type=str, default='outputs')
             
    args = parser.parse_args()
    sr = 16000

    print(f'Load codec ckpt from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    encoder = CodecEncoder()
    encoder.load_state_dict(ckpt['CodecEnc'])
    encoder = encoder.eval().cuda()
    decoder = CodecDecoder()
    decoder.load_state_dict(ckpt['generator'])
    decoder = decoder.eval().cuda()

    wav_dir = args.output_dir
    os.makedirs(wav_dir, exist_ok=True)
    wav_paths = glob(join(args.input_dir, '*.wav'))
    print(f'Found {len(wav_paths)} wavs in {args.input_dir}')
    
    st = time()
    for wav_path in tqdm(wav_paths):
        target_wav_path = join(wav_dir, basename(wav_path))
        wav = librosa.load(wav_path, sr=sr)[0] 
        wav = torch.from_numpy(wav).unsqueeze(0).cuda()
        wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
        with torch.no_grad():
            vq_emb = encoder(wav.unsqueeze(1))
            vq_post_emb, vq_code, _ = decoder(vq_emb, vq=True)
            recon = decoder(vq_post_emb, vq=False).squeeze().detach().cpu().numpy()
        sf.write(target_wav_path, recon, sr)
    et = time()
    print(f'Inference ends, time: {(et-st)/60:.2f} mins')
