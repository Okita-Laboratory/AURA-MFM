# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import os
import sys
import glob
import argparse
import torch

class Logger:
    def __init__(self, ckpt_path):
        self.eval_dir = ckpt_path.replace(ckpt_path.split('/')[-1], 'eval')
        os.makedirs(self.eval_dir, exist_ok=True)
        
        self.file = open(os.path.join(self.eval_dir, 'out.log'), 'w')
    def write(self, msg):
        self.file.write(msg)
    def flush(self):
        self.file.flush()


def main(args):
    ##############
    # Set up meta
    ##############
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger = Logger(args.ckpt_path)
    sys.stdout = logger
    
    print('\n## Configuration: ')
    for k, v in vars(args).items():
        print(k, v)

    
    ###############
    # Set up model
    ###############
    from lib.models import MW2StackRNNPooling, MW2StackRNNPoolingMocap, CrossModalContrastiveLearningModule
    from lib.clip_model import ClipPLModel
    import lib.transformer as Ts
    state_dict = torch.load(args.ckpt_path)
    ckpt_args = state_dict['args']

    encoders = {}
    if 'imu' in ckpt_args.modality:
        imu_encoder = (
            MW2StackRNNPooling(size_embeddings=ckpt_args.size_embedding)                                            if ckpt_args.arch == 'rnn' else
            Ts.__dict__[ckpt_args.model_size](window_size=1000, in_chans=6, patch_size=ckpt_args.patch_size['imu']) if ckpt_args.arch == 'transformer' else None
        )
        encoders['imu'] = imu_encoder
    if 'mocap' in ckpt_args.modality:
        mocap_encoder = (
            MW2StackRNNPoolingMocap(size_embeddings=ckpt_args.size_embedding)                                        if ckpt_args.arch == 'rnn' else
            Ts.__dict__[ckpt_args.model_size](window_size=50, in_chans=51, patch_size=ckpt_args.patch_size['mocap']) if ckpt_args.arch == 'transformer' else None
        )
        encoders['mocap'] = mocap_encoder
    if 'text' in ckpt_args.modality:
        text_encoder = ClipPLModel(freeze=True, device=device).to(dtype=torch.float)
        encoders['text'] = text_encoder
    
    model = CrossModalContrastiveLearningModule(encoders, ckpt_args.modality, device=device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    

    ##############
    # Set up data
    ##############
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from lib.dataloader import CrossModalDataset, worker_init_fn
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    dataset = CrossModalDataset(path=args.data_path, modality=ckpt_args.modality, name='test', transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn, shuffle=True, pin_memory=True)

    
    from lib.evaluation import evaluate
    target = 'mocap' if 'mocap' in ckpt_args.modality else 'text'
    metrics = evaluate(loader, model, source='imu', target=target, result_path=logger.eval_dir, device=device)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,  default='/data4/dataSpace/EGO_EXO4D/preprocessed/datasets/5s_datafiles')
    parser.add_argument("--ckpt-path", type=str, default='<best model path>')
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
