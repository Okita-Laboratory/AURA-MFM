import os
import json
import numpy as np
from tqdm.auto import tqdm

FPS = 30
HZ = 200
SECOND = 5.0
IMU_LENGTH = int(SECOND * HZ)


dataset_root_dir = '/data4/dataSpace/EGO_EXO4D/'
captures = json.load(open(os.path.join(dataset_root_dir, 'captures.json')))
takes = json.load(open(os.path.join(dataset_root_dir, 'takes.json')))
comments_train_json = json.load(open(os.path.join(dataset_root_dir, 'annotations/expert_commentary_train.json')))['annotations']
comments_val_json = json.load(open(os.path.join(dataset_root_dir, 'annotations/expert_commentary_val.json')))['annotations']

dir = '/data4/dataSpace/EGO_EXO4D/preprocessed'
imu_dir = f'{dir}/imu_right_200hz'
# imu_dir = f'{dir}/imu_left_200hz'


path = f'/home/ukita/data4/dataSpace/EGO_EXO4D/preprocessed/datasets/imu_text'
os.makedirs(path, exist_ok=True)

for i, take in tqdm(enumerate(takes), total=len(takes)):
    imu_path = f'{imu_dir}/{i:04}_imu.npy'
    if not os.path.exists(imu_path):
        continue
    imu = np.load(imu_path)

    take_uid = take['take_uid']
    try:
        comments_json = comments_train_json[take_uid]
    except KeyError:
        try:
            comments_json = comments_val_json[take_uid]
        except KeyError:
            continue
    comments = comments_json[0]['commentary_data']
    task = comments_json[0]['task_name']

    for n, comment in enumerate(comments):
        idx = int(comment['recording'][:-5])
        text = f"The following text was annotated by an expert after watching the video from the start of task '{task}' up to {comment['video_time']} seconds: {comment['text']}"

        imu_end_idx = int(comment['video_time']*HZ)+1
        if imu_end_idx - IMU_LENGTH > 0:
            imu_window = imu[imu_end_idx-IMU_LENGTH:imu_end_idx]
            np.save(f'{path}/{i:04}_{idx:02}_imu.npy', imu_window)
            with open(f'{path}/{i:04}_{idx:02}_text.txt', 'w') as f:
                f.write(text)
