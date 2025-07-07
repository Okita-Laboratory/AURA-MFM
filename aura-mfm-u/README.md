
# Installation

```
conda create -n imuclip python=3.8 pip git
conda activate imuclip
pip install pytorch_lightning==1.9.1
pip install torch==1.13.1
pip install torchaudio==0.13.1
pip install torchvision==0.14.1
pip install torchmetrics==0.11.1
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
pip install matplotlib
pip install ffmpeg-python
pip install pandas
```

# pre-trained model
明確に訓練済みモデルとして公開されているものはなかったが、このimu2clipの貢献者の一人のpranayさんがhuggingfaceで公開していたので、おそらくこれだろうというモデルを発見した。
モデルカードの記載がなく、７つの似たようなモデルが置かれていただけなので、もしかすると全く違うモデルかもしれない。
```bash
wget https://huggingface.co/pranay-ar/IMU2CLIP/resolve/main/mw2/i2c/egohos_best.pt
```

# Experiments
**To run an example train loop**
```
python pretraining.py
```

**To run a pretrained model in downstream task**
```
python downstream.py
```

In the config folder, you can find details hyperparamters for training IMU2CLIP with different contrastive losses. 

## clip
```python
import clip
print(clip.available_models())
model, preprocess = clip.load("ViT-B/32")
print(f'token vocab      : {model.vocab_size}')
print(f'token length     : {model.context_length}')
print(f'image resolution : {model.visual.input_resolution}')
```
```
['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
token vocab      : 49408
token length     : 77
image resolution : 224
```

```python
import torch
from PIL import Image

device = 'cuda:0'
model, preprocess = clip.load('ViT-B/32', device=device)

# text
text = 'sample sentence'
text_tokens = clip.tokenize(text).to(device)
text_features = model.encode_text(text_tokens)

# image
image = Image.open('sample.png')
image_tensor = preprocess(image).unsqueeze(0).to(device)
image_features = model.encode_image(image_tensor)

print(text_features.shape, image_features.shape)
```
```
torch.Size([1, 512]) torch.Size([1, 512])
```

## IMU2CLIP
```python
import torch
from lib.train_modules import MultimodalContrastiveLearningModule
from lib.loss import InfoNCE
from lib.clip_model import ClipPLModel
from lib.imu_models import MW2StackRNNPooling

# using ViT B/32
imu_encoder   = MW2StackRNNPooling(size_embeddings=512)
text_encoder  = ClipPLModel(freeze=True)
video_encoder = ClipPLModel(freeze=True)
video_encoder.video_encoder_name = 'clip_1frame' # 'clip_1frame' or 'clip_avg_frames'

model = MultimodalContrastiveLearningModule(
    modality_to_encoder={
        'text': text_encoder,
        'imu': imu_encoder,
        'video': video_encoder
    },
    source_modality='imu',
    target_modalities=['text', 'video'],
)

# pre-trained model from "https://huggingface.co/pranay-ar/IMU2CLIP/resolve/main/mw2/i2c/egohos_best.pt"
state_dict = torch.load('./saved/i2c/egohos_best.pt') 
model.load_state_dict(state_dict, strict=False)


B = 1
# IMU process
x_imu = torch.randn(B, 6, 1000) # [batch, channels, length]
out_imu = model.imu_encoder(x_imu)

# text process
x_narration = ['running on the road' for _ in range(B)] # [batch, text(any length)]
out_narration = model.text_encoder.get_text_embeddings(x_narration)

# video process
x_video = torch.randn(B, 3, 10, 224, 224) # [batch, channels, frames, width, height]
out_video = model.video_encoder.get_video_embeddings(x_video)


from lib.loss import InfoNCE
loss = InfoNCE(symmetric_loss=True, learn_temperature=True)
loss_output = 0.0

source_modality = out_imu

for target_modality in [out_narration, out_video]:
    s2t_loss = loss(query=source_modality, positive_key=target_modality)
    loss_output += s2t_loss
```