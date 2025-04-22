# AURA-MFM
現状のコードは[IMU2CLIP](https://github.com/facebookresearch/imu2clip)が基になっています。
## 環境構築
```
conda create -n aura-mfm python=3.8
conda activate aura-mfm
pip install -r req.txt
```

## 実験
**imuとvideoのペアによる対比学習**
```
python pretraining.py　--path_configs ./configs/train_contrastive/ego4d_imu2video.yaml
```

**imuとtextのペアによる対比学習**
```
python pretraining.py　--path_configs ./configs/train_contrastive/ego4d_imu2text.yaml
```
基本的な設定はyaml内で可能です。
+ 主な変更点
  * `imu_encoder_name` : "senvt"または"mw2"(imu2clip)を指定
  * `path_load_from_checkpoint` : チェックポイント(.ckpt)の読み込み
  * `use_egohos_best_pt` : imu2clipの事前学習モデル(/home/matsuishi/aura-mfm/egohos_best.pt)の読み込み
  * `model_size` : SENvTのモデルサイズ
  * `patch_size` : SENvTのパッチサイズ
  * `data_path` : ego-exo4dデータのパス (checkpoint/full_videos)
