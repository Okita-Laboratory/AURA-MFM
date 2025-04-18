import csv
import glob

import tqdm

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import RecordableTypeId

for vrsfile in tqdm.tqdm(glob.glob(f"/data4/dataSpace/EGO_EXO4D/takes/*/*.vrs")):
    vrs_data_provider = data_provider.create_vrs_data_provider(vrsfile)
    name = vrsfile.split("/")[-2]
    if not vrs_data_provider:
        print("Invalid vrs data provider")

    # CSVファイルを書き込みモードでオープン
    with open(
        f"ego-exo4d-sample/imu/{name}.csv", "w", newline=""
    ) as csvfile:
        # CSVライターの設定
        csvwriter = csv.writer(csvfile)
        # ヘッダーの書き込み
        headers = ["timestamp_ms", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        csvwriter.writerow(headers)

        stream_id = vrs_data_provider.get_stream_id_from_label("imu-right")
        for index in range(0, int(vrs_data_provider.get_num_data(stream_id))):
            imu_data = vrs_data_provider.get_imu_data_by_index(stream_id, index)

            # タイムスタンプをnsからmsに変換
            timestamp_ms = imu_data.capture_timestamp_ns / 1e6

            # データ行の書き込み
            row = [
                timestamp_ms,
                imu_data.accel_msec2[0],
                imu_data.accel_msec2[1],
                imu_data.accel_msec2[2],
                imu_data.gyro_radsec[0],
                imu_data.gyro_radsec[1],
                imu_data.gyro_radsec[2],
            ]
            csvwriter.writerow(row)
