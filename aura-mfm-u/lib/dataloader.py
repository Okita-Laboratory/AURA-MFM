import os
import glob
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def worker_init_fn(worker_id):
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def collate_fn(batch):
    pass

class ImuMocapDataset(Dataset):
    def __init__(self, path, name='train', transform=None, random_split=False):
        self.transform = transform
        if random_split:
            if 'train' in name:
                self.files = sorted(np.loadtxt(os.path.join(path, 'train_files.txt'), dtype=str))
            elif 'val' in name:
                self.files = sorted(np.loadtxt(os.path.join(path, 'val_files.txt'), dtype=str))
            elif 'test' in name:
                self.files = sorted(np.loadtxt(os.path.join(path, 'test_files.txt'), dtype=str))
            else:
                print('error: unkown data type')
            self.imu_files = [file+'_imu.npy' for file in self.files]
            self.mocap_files = [file+'_mocap.npy' for file in self.files]
        else:
            name = 'train' if 'train' in name else 'val'
            self.imu_files = sorted(glob.glob(os.path.join(path, name, '*_imu.npy')))
            self.mocap_files = sorted(glob.glob(os.path.join(path, name, '*_mocap.npy')))

    def __len__(self):
        return len(self.mocap_files)
    
    def __getitem__(self, idx):
        imu = np.load(self.imu_files[idx]).astype(np.float32)
        mocap = np.load(self.mocap_files[idx]).astype(np.float32)
        
        if self.transform is not None:
            imu = self.transform(imu)
            imu = imu.squeeze(0) if imu.size(0) == 1 else imu
            
            mocap = self.transform(mocap)
            mocap = mocap.squeeze(0) if mocap.size(0) == 1 else mocap
            # ここでnanが発生するのは、部位欠損だと思われるので(それ以外は線形補間済み)、0埋めにする(0埋めでいいのかはわからない。)
            if torch.isnan(mocap).any(): 
                mocap = torch.nan_to_num(mocap)
        
        return dict(imu=imu, mocap=mocap)


class ImuTextDataset(Dataset):
    def __init__(self, path, name='train', transform=None):
        self.transform = transform
        if 'train' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'train_files.txt'), dtype=str))
        elif 'val' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'val_files.txt'), dtype=str))
        elif 'test' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'test_files.txt'), dtype=str))
        else:
            print('error: unkown data type')
        self.imu_files = [file+'_imu.npy' for file in self.files]
        self.text_files = [file+'_text.txt' for file in self.files]

    def __len__(self):
        return len(self.imu_files)
    
    def __getitem__(self, idx):
        imu = np.load(self.imu_files[idx]).transpose(1, 0).astype(np.float32)
        text = self.load_text(self.text_files[idx])
        
        if self.transform is not None:
            imu = self.transform(imu)
            imu = imu.squeeze(0) if imu.size(0) == 1 else imu
        return dict(imu=imu, text=text)
    
    def load_text(self, path):
        with open(path, 'r') as f:
            text = f.read()
        return text


class CrossModalDataset(Dataset):
    def __init__(self, path, modality, name='train', transform=None):
        self.transform = transform
        if 'train' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'train_files.txt'), dtype=str))
        elif 'val' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'val_files.txt'), dtype=str))
        elif 'test' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'test_files.txt'), dtype=str))
        else:
            print('error: unkown data type')
        
        self.modality = modality
        if 'imu' in self.modality:
            self.imu_files = [file+'_imu.npy' for file in self.files]
        if 'mocap' in self.modality:
            self.mocap_files = [file+'_mocap.npy' for file in self.files]
        if 'text' in self.modality:
            self.text_files = [file+'_text.txt' for file in self.files]

    def __len__(self):
        return len(self.imu_files)
    
    def __getitem__(self, idx):
        batch = {}
        if 'imu' in self.modality:
            # imu = np.load(self.imu_files[idx]).transpose(1, 0).astype(np.float32)
            imu = np.load(self.imu_files[idx]).astype(np.float32)
            if self.transform is not None:
                imu = self.transform(imu)
                imu = imu.squeeze(0) if imu.size(0) == 1 else imu
            batch['imu'] = imu
        
        if 'mocap' in self.modality:
            mocap = np.load(self.mocap_files[idx]).astype(np.float32)
            if self.transform is not None:
                mocap = self.transform(mocap)
                mocap = mocap.squeeze(0) if mocap.size(0) == 1 else mocap
                # ここでnanが発生するのは、部位欠損だと思われるので(それ以外は線形補間済み)、0埋めにする(0埋めでいいのかはわからない。)
                if torch.isnan(mocap).any():
                    mocap = torch.nan_to_num(mocap)
            batch['mocap'] = mocap
        
        if 'text' in self.modality:
            text = self.load_text(self.text_files[idx])
            batch['text'] = text

        return batch
    
    def load_text(self, path):
        with open(path, 'r') as f:
            text = f.read()
        return text



class ImuClassifyDataset(Dataset):
    def __init__(self, path, name='train', transform=None):
        self.transform = transform
        if 'train' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'train_files.txt'), dtype=str))
        elif 'val' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'val_files.txt'), dtype=str))
        elif 'test' in name:
            self.files = sorted(np.loadtxt(os.path.join(path, 'test_files.txt'), dtype=str))
        else:
            print('error: unkown data type')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        imu, label = np.load(self.files[idx], allow_pickle=True)
        imu = imu.astype(np.float32)
        
        if self.transform is not None:
            imu = self.transform(imu)
            imu = imu.squeeze(0) if imu.size(0) == 1 else imu
        return imu, label
