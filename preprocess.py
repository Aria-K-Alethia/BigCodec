import os
import hydra
import librosa
import utils
from os.path import expanduser, exists, basename, join
from utils import read_filelist, write_filelist, find_all_files
from tqdm import tqdm

@hydra.main(version_base=None, config_path='config', config_name='default')
def preprocess(cfg):
    os.makedirs('filelists', exist_ok=True)
    # train
    root = cfg.preprocess.datasets.LibriSpeech.root
    root = expanduser(root)
    trainfiles = []
    print(f'Root: {root}')
    for subset in cfg.preprocess.datasets.LibriSpeech.trainsets:
        files = find_all_files(join(root, subset), '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        for i in range(len(files)):
            files[i][1] = files[i][1].replace(root, '').lstrip('/')
        trainfiles.extend(files)
    
    print(f'Write train filelist to {cfg.preprocess.view.train_filelist}')
    os.makedirs('filelists', exist_ok=True)
    utils.write_filelist(trainfiles, cfg.preprocess.view.train_filelist)

if __name__ == '__main__':
    preprocess()
