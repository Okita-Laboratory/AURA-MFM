import glob
import json
import random

import numpy as np
import tqdm

# JSONファイルの読み込み
json_path = "takes.json"
with open(json_path, "r", encoding="utf-8") as f_name:
    json_data = json.load(f_name)

# take_listの生成
take_list = [take for take in json_data if take["has_trimmed_vrs"]]
take_name_list = []
for t in tqdm.tqdm(take_list):
    timestamp = np.load(
        "../checkpoint/full_videos/processed_imu/{}_timestamps.npy".format(
            t["take_name"]
        )
    )

    if not (timestamp[-1] / 1000 < t["duration_sec"] - 1):
        take_name_list.append(t["take_name"])
    else:
        print(timestamp[-1] / 1000, t["duration_sec"])
# take_list = [
#     p.split("/")[-1].replace("_timestamps.npy", "")
#     for p in glob.glob("../checkpoint/full_videos/processed_imu/*_timestamps.npy")
# ]
# データをシャッフル
random.seed(42)  # 再現性のため固定のシード値を設定
random.shuffle(take_name_list)
take_name_list = take_name_list[:1000]

# データの分割
total = len(take_name_list)
train_end = int(total * 0.8)
valid_end = int(total * 0.9)

train_list = take_name_list[:train_end]
valid_list = take_name_list[train_end:valid_end]
test_list = take_name_list[valid_end:]

# 各リストをJSONファイルとして保存
output_dir = "../../splits_exo/"
with open(output_dir + "training_2.json", "w", encoding="utf-8") as train_file:
    json.dump(train_list, train_file, ensure_ascii=False, indent=4)
with open(output_dir + "validation_2.json", "w", encoding="utf-8") as valid_file:
    json.dump(valid_list, valid_file, ensure_ascii=False, indent=4)
with open(output_dir + "test_2.json", "w", encoding="utf-8") as test_file:
    json.dump(test_list, test_file, ensure_ascii=False, indent=4)

print("データ分割と保存が完了しました。")
