import os
import json
import numpy as np
from tqdm.auto import tqdm
from utils import get_mocap

dataset_root_dir = '/data4/dataSpace/EGO_EXO4D/'
captures = json.load(open(os.path.join(dataset_root_dir, 'captures.json')))
takes = json.load(open(os.path.join(dataset_root_dir, 'takes.json')))

dim = '3D'
train_or_val = 'val'

path = f'/data4/dataSpace/EGO_EXO4D/preprocessed/motion_capture/{dim}/{train_or_val}/'
os.makedirs(path, exist_ok=True)

for i, take in tqdm(enumerate(takes), total=len(takes)):
    try:
        frame_nums, mocaps = get_mocap(take, dim=dim, train_or_val=train_or_val)
        np.save(f'{path}{i:04}_motion_capture.npy', mocaps)
        np.save(f'{path}{i:04}_frame_number.npy', frame_nums)
    except:
        continue