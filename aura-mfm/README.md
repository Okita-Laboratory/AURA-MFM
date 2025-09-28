# AURA-MFM
## Environment Setup
```
conda create -n aura-mfm python=3.8
conda activate aura-mfm
pip install -r req.txt
```

## Experiments
**Contrastive learning between IMU and video**
```
python pretraining.py --path_configs ./configs/train_contrastive/ego4d_imu2video.yaml
```

**Contrastive learning between IMU and text**
```
python pretraining.py --path_configs ./configs/train_contrastive/ego4d_imu2text.yaml
```
Most settings can be configured in the corresponding YAML files.
+ Key configuration options
  * `imu_encoder_name`: Specify `"senvt"` or `"mw2"` (used in IMU2CLIP)
  * `path_load_from_checkpoint`: Path to load a `.ckpt` checkpoint
  * `path_load_pretrained_imu_encoder`: Path to load a `.pt` encoder weights
  * `use_egohos_best_pt`: Load the pretrained model `egohos_best.pt` from IMU2CLIP
  * `model_size`: Model size for SENvT
  * `patch_size`: Patch size for SENvT
  * `data_path`: Path to the Ego-Exo4D data (e.g., `checkpoint/full_videos`)

To perform retrieval tasks, set `test_only` to `True` and specify the path to the checkpoint in `path_load_from_checkpoint`.
