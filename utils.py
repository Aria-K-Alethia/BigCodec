import os
from os.path import join, exists, basename
from pathlib import Path 

def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append([(str(Path(f).stem)), os.path.join(root, f)])
    return out

def read_filelist(path, delimiter='|'):
    with open(path) as f:
        lines = [line.strip().split(delimiter) for line in f if line.strip()]
    return lines

def write_filelist(filelists, path, delimiter='|'):
    with open(path, 'w', encoding='utf8') as f:
        for line in filelists:
            f.write(delimiter.join(line) + '\n')
