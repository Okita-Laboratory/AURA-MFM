# Ego-Exo4D

`kobu:/data4/dataSpace/EGO_EXO4D`<br>`shinmachi:/data4/dataSpace/EGO_EXO4D`

[Ego-Exo4D](https://docs.ego-exo4d-data.org/)データセットをダウンロードしました。

'''
Ego-Exo4D は、大規模なマルチモーダル マルチビュー ビデオ データセット (3D を含む) およびベンチマーク チャレンジです。データセットは、少なくとも 1 つの一人称 (エゴセントリック Aria グラス) および三人称 (エキソセントリック GoPro カメラ) のパースペクティブ カメラで録画された参加者の時間同期ビデオで構成されています。録画は、世界中の 12 の異なる研究機関で行われました。各録画 (キャプチャ) には、1 人以上の参加者 (カメラ着用者) が物理的なタスク (サッカー、バスケットボール、ダンス、ボルダリング、音楽) または手順的なタスク (料理、自転車修理、健康) を実行する複数のテイクが含まれています。Aria グラスを使用しているため、関連する 3D データが広範囲に存在します。'''<br>
(引用：https://docs.ego-exo4d-data.org/overview/)

## download
- `aws-cli`を使用してダウンロードしています。
- [ライセンス](https://ego4ddataset.com/egoexo-license/)への同意とdropbox経由の署名(10箇所くらい)により、ちょうど2日で承認され`Access ID`と`Access Key`を受け取ることができます。
- `aws configure`というコマンドで送信されてきた以上二つのパラメータを入力して設定完了です。
- `Access ID`と`Access Key`は受信してから14日で切れるので14が過ぎた後に追加でデータをダウンロードしたい時は再度、同意・署名を行う必要があります。
```bash
egoexo -o /data4/dataSpace/EGO_EXO4D --parts default
```
- 以上のコマンドを叩いて約2-3日ほどでダウンロードが完了しました。
- defaultは以下のpartを含みます。詳細は[ここ](https://docs.ego-exo4d-data.org/download/)。
- default(12.112 TB)
    - [`metadata`](https://docs.ego-exo4d-data.org/data/metadata/)(46 MB)
    - [`annotations`](https://docs.ego-exo4d-data.org/annotations/)(10.533 GB)
    - [`takes`](https://docs.ego-exo4d-data.org/data/takes/)(10.553 TB)
    - `captures`(43.618 GB)
    - [`take_trajectory`](https://docs.ego-exo4d-data.org/data/mps/#trajectory)(509.503 GB)
    - `take_vrs_noimagestream`(995.592 GB)
- 加えて短辺が448pxに縮小されたテイクもダウンロードしました。<br>[`downscaled_takes/448`](https://docs.ego-exo4d-data.org/data/downscaled_takes/)(438.556 GB)
```bash
egoexo -o /data4/dataSpace/EGO_EXO4D --parts downscaled_takes/448
```

## Data
全ての詳細を書くことはしない。データを触る前に事前に知っておきたかったことを軽くまとめる。
より詳しく知りたい場合は[ここ](https://docs.ego-exo4d-data.org/data/)やデータを提供している[ここ](https://github.com/facebookresearch/Ego4d/issues)でissueを立ててみる、もしくは直接メールをするといいだろう。
- `takes.json`: 各データはtakeと呼ばれる単位で保存されており、計5035のtakeがある。(ちなみに撮影された動画が連なったものはcaptureと呼ばれ、takeは各captureの中から切り取ったものである。captureは計787ある。)一般的に、この`takes.json`を読み込み各データにアクセする形を取る。
    ```python
    root_dir = '/data4/dataSpace/EGO_EXO4D'
    takes = json.load(open(f'{root_dir}/takes.json'))
    takes
    ------------------- output -------------------
    [{
        'root_dir': 'takes/cmu_bike01_2',
        'take_name': 'cmu_bike01_2',
        'participant_uid': 657,
        'is_dropped': False,
        'objects': [],
        'task_id': 4001,
        'task_name': 'Remove a Wheel',
        'parent_task_id': 4000,
        'parent_task_name': 'Bike Repair',
        'take_idx': 2,
        'duration_sec': 110.93333333333334,
        'best_exo': 'cam04',
        'task_start_sec': 0.28164,
        'task_end_sec': 110.06668,
        'task_timing_annotation_uid': '543a6b43-73ee-41d0-a45e-dbb8ce12434d',
        'is_narrated': False,
        'capture_uid': 'd37b73eb-fa42-43a6-8115-56832996ebd7',
        'take_uid': 'ed3ec638-8363-4e1d-9851-c7936cbfad8c',
        'take_timing_uid': '7811bafb-e625-4bd4-ab50-bf61c18e57c1',
        'timesync_uid': 'f66bd964-1bc1-4da5-a55f-4d5fc770db60',
        'timesync_start_idx': 7170,
        'timesync_end_idx': 10498,
        'frame_aligned_videos': {'aria01': {'slam-left': {'cam_id': 'aria01',
            'stream_id': '1201-1',
            'readable_stream_id': 'slam-left',
        ...
        'timesync_validated': True,
        'validated': True},
        'physical_setting_uid': 78,
        'validated': True},
    ...]
    ```
