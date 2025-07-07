import os
import json
import numpy as np
from tqdm.auto import tqdm

FPS = 30
HZ = 200
SECOND = 5.0
MOCAP_LENGTH = int(SECOND * FPS/3)
IMU_LENGTH = int(SECOND * HZ)
THRESHOLD = 0.9 * MOCAP_LENGTH
CHANNELS = int(17 * 3)

def next(array):
    i = 0
    while i < len(array)-1:
        yield i, (array[i], array[i+1])
        i += 1

def get_mocap_and_imu(fn, mocap, imu):
    mocap_start_idx = 0
    mocap_units, imu_units = [], []
    for i, (nc, nn) in next(fn):
        mocap_start_sec = fn[mocap_start_idx]/FPS
        converted_imu_start_idx = int(mocap_start_sec*HZ)
        length = i - mocap_start_idx
        if (nn - nc > MOCAP_LENGTH) or (length == MOCAP_LENGTH):
            if length < THRESHOLD:
                mocap_start_idx = i+1
                continue

            imu_unit = imu[converted_imu_start_idx:converted_imu_start_idx+IMU_LENGTH].transpose(1, 0)
            if len(imu_unit[0]) < IMU_LENGTH:
                continue
            imu_units.append(imu_unit)
            
            mocap_unit = np.zeros((CHANNELS, MOCAP_LENGTH))
            mocap_unit_ = mocap[:, mocap_start_idx:i]
            mocap_unit[:, :length] = mocap_unit_
            
            delta = int(MOCAP_LENGTH - length)
            if delta < 0:
                mocap_unit[:, -1*delta:] = np.repeat(mocap_unit_[:, -1], delta).reshape(CHANNELS, delta)
            mocap_units.append(mocap_unit)
            mocap_start_idx = i+1
    return mocap_units, imu_units


dataset_root_dir = '/data4/dataSpace/EGO_EXO4D/'
captures = json.load(open(os.path.join(dataset_root_dir, 'captures.json')))
takes = json.load(open(os.path.join(dataset_root_dir, 'takes.json')))

train_or_val = 'val'

dir = '/data4/dataSpace/EGO_EXO4D/preprocessed'
mocap_dir = f'{dir}/motion_capture/3D/{train_or_val}'
imu_dir = f'{dir}/imu_right_200hz'
# imu_dir = f'{dir}/imu_left_200hz'


path = f'/data4/dataSpace/EGO_EXO4D/preprocessed/datasets/{train_or_val}'
os.makedirs(path, exist_ok=True)

for i, take in tqdm(enumerate(takes), total=len(takes)):
    mocap_path = f'{mocap_dir}/{i:04}_motion_capture.npy'
    imu_path = f'{imu_dir}/{i:04}_imu.npy'
    if os.path.exists(mocap_path) and os.path.exists(imu_path):
        fn = np.load(f'{mocap_dir}/{i:04}_frame_number.npy')
        mocap = np.load(f'{mocap_dir}/{i:04}_motion_capture.npy')

        imu = np.load(f'{imu_dir}/{i:04}_imu.npy')

        mocaps, imus = get_mocap_and_imu(fn, mocap, imu)
        for j in range(len(mocaps)):
            np.save(f'{path}/{i:04}_{j:02}_mocap.npy', mocaps[j])
            np.save(f'{path}/{i:04}_{j:02}_imu.npy', imus[j])
