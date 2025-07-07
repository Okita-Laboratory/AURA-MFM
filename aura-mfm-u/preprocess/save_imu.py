import os
import json
import numpy as np
from tqdm.auto import tqdm
from utils import get_imu

dataset_root_dir = '/data4/dataSpace/EGO_EXO4D/'
captures = json.load(open(os.path.join(dataset_root_dir, 'captures.json')))
takes = json.load(open(os.path.join(dataset_root_dir, 'takes.json')))


part = 'right'
hz = 1000

path = f'/data4/dataSpace/EGO_EXO4D/preprocessed/imu_{part}_{hz}hz/'
os.makedirs(path, exist_ok=True)

for i, take in tqdm(enumerate(takes), total=len(takes)):
    try:
        timestamps, imus = get_imu(take, imu_part=f'imu-{part}', hz=hz)
        np.save(f'{path}{i:04}_{take["take_name"]}_imu.npy', imus)
    except:
        continue