# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import os
import sys
import glob
import argparse
import copy
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Logger:
    def __init__(self, ckpt_path, debug=False):
        if debug:
            self.file = sys.stdout
        else:
            self.eval_dir = ckpt_path.replace(ckpt_path.split('/')[-1], 'downstream')
            os.makedirs(self.eval_dir, exist_ok=True)
            self.file = open(os.path.join(self.eval_dir, 'out.log'), 'w')
    def write(self, msg):
        self.file.write(msg)
    def flush(self):
        self.file.flush()

def train_model(model, train_loader, valid_loader, optimizer, loss_fn, epochs, device='cpu'):
    train_losses, train_acces, train_f1s = [], [], []
    valid_losses, valid_acces, valid_f1s = [], [], []
    best_acc, best_f1 = 0., 0.
    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, running_f1 = [], [], []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            torch.autograd.set_detect_anomaly(True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pred_y = torch.argmax(pred, dim=1)
            train_acc = accuracy_score(y.cpu(), pred_y.cpu())
            train_f1 = f1_score(y.cpu(), pred_y.cpu(), average='macro')

            running_loss.append(loss.item())
            running_acc.append(train_acc)
            running_f1.append(train_f1)
        
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, valid_loader, loss_fn, device=device)
        
        train_losses.append(np.mean(running_loss))
        train_acces.append(np.mean(running_acc))
        train_f1s.append(np.mean(running_f1))
        valid_losses.append(val_loss)
        valid_acces.append(val_acc)
        valid_f1s.append(val_f1)
        
        if best_f1 <= val_f1:
            best_f1 = val_f1
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
            update_msg = 'Update: f-score'
            if best_acc <= val_acc:
                best_acc = val_acc
                update_msg += ' & accuracy'
        elif best_acc <= val_acc:
            best_acc = val_acc
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
            update_msg = 'Update: accuracy'
        else:
            update_msg = ''
        
        epoch_len = len(str(epochs))
        print_msg = (
            f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] '
            + f'Loss: [train: {np.mean(running_loss):.4f}, valid: {val_loss:.4f}] '
            + f'Accuracy: [train: {np.mean(running_acc):.4f}, valid: {val_acc:.4f}] '
            + f'F-score: [train: {np.mean(running_f1):.4f}, valid: {val_f1:.4f}] '
            + update_msg
        )
        print(print_msg)
    log = {
        'loss': {'train': train_losses, 'valid': valid_losses},
        'acc' : {'train': train_acces, 'valid': valid_acces}
    }
    return log, best

def evaluate_model(model, data_loader, loss_fn, is_zeroshot=False, device='cpu'):
    model.eval()
    losses, trues, preds = [], [], []
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model.zeroshot_classification(x) if is_zeroshot else model(x)
            loss = loss_fn(pred, y)
            pred_y = torch.argmax(pred, dim=1)
        
        losses.append(loss.item())
        trues.append(y.detach().cpu())
        preds.append(pred_y.detach().cpu())
    trues = torch.cat(trues)
    preds = torch.cat(preds)
    return (
        np.mean(np.array(losses)),
        accuracy_score(trues, preds),
        f1_score(trues, preds, average='macro'),
        trues,
        preds,
    )

def save_fig(log, path):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,4))

    fig.add_subplot(1,2,1)
    plt.plot(log['loss']['train'], label='train')
    plt.plot(log['loss']['valid'], label='val')
    plt.title('loss')
    plt.legend()

    fig.add_subplot(1,2,2)
    plt.plot(log['acc']['train'], label='train')
    plt.plot(log['acc']['valid'], label='val')
    plt.title('accuracy')
    plt.legend()

    plt.savefig(path)

def freeze(params):
    params.eval()
    for p in params.parameters():
        p.requires_grad = False



def main(args):
    ##############
    # Set up meta
    ##############
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger = Logger(args.ckpt_path, debug=args.debug)
    sys.stdout = logger
    
    print('\n## Configuration: ')
    for k, v in vars(args).items():
        print(k, v)
    

    ##############
    # Set up data
    ##############
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from lib.dataloader import ImuClassifyDataset, worker_init_fn
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])

    train_dataset = ImuClassifyDataset(path=args.data_path, name='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn, shuffle=True, pin_memory=True)

    val_dataset = ImuClassifyDataset(path=args.data_path, name='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn, shuffle=False, pin_memory=True)

    test_dataset = ImuClassifyDataset(path=args.data_path, name='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn, shuffle=False, pin_memory=True)

    
    ###############
    # Set up model
    ###############
    state_dict = torch.load(args.ckpt_path)
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    if 'imu_encoder.net.0.weight' in state_dict.keys():
        from lib.models import MW2StackRNNPooling
        imu_encoder = MW2StackRNNPooling(size_embeddings=512)
    elif 'imu_encoder.layers.0.norm1.weight' in state_dict.keys():
        import lib.transformer as Ts
        imu_encoder = Ts.__dict__['SS'](window_size=1000, in_chans=6, patch_size=8)
    elif 'net.0.weight' in state_dict.keys():
        from lib.models import MW2StackRNNPooling
        imu_encoder = MW2StackRNNPooling(size_embeddings=512)
        imu_encoder.load_state_dict(state_dict, strict=False)
    elif 'layers.0.norm1.weight' in state_dict.keys():
        import lib.transformer as Ts
        imu_encoder = Ts.__dict__['SS'](window_size=1000, in_chans=6, patch_size=8)
        imu_encoder.load_state_dict(state_dict, strict=False)
    else:
        print('error')
        sys.exit()
    
    from lib.models import IMUClasifyModule
    model = IMUClasifyModule(imu_encoder, size_embeddings=512, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    ##################
    # Set up training
    ##################
    loss_fn = nn.CrossEntropyLoss()
    
    
    # zero-shot #
    if args.downstream == 'zeroshot':
        _, acc, f1, _, _ = evaluate_model(model, test_loader, loss_fn, is_zeroshot=True, device=device)
        print(f'acc: {acc:.6f}, f1: {f1:.6f}')
    
    
    # transfer learning #
    elif args.downstream == 'transfer':
        freeze(model.imu_encoder)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        log, best = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=args.epochs, device=device)

        if not args.debug:
            torch.save(best['model'].state_dict(), os.path.join(logger.eval_dir, f'best_{best["epoch"]}.pt'))
            save_fig(log, os.path.join(logger.eval_dir, 'loss.png'))
        
        _, acc, f1, _, _ = evaluate_model(best['model'], test_loader, loss_fn, device=device)
        print('### test ###')
        print(f'acc: {acc:.6f}, f1: {f1:.6f}')
    
    # fine-tuning #
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        log, best = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=args.epochs, device=device)

        if not args.debug:
            torch.save(best['model'].state_dict(), os.path.join(logger.eval_dir, f'best_{best["epoch"]}.pt'))
            save_fig(log, os.path.join(logger.eval_dir, 'loss.png'))
        
        _, acc, f1, _, _ = evaluate_model(best['model'], test_loader, loss_fn, device=device)
        print('### test ###')
        print(f'acc: {acc:.6f}, f1: {f1:.6f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--data-path", type=str,  default='/home/ukita/data4/dataSpace/EGO_EXO4D/preprocessed/downstream/splits_files')
    parser.add_argument("--ckpt-path", type=str, default='<best model path>')
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--downstream", type=str, default='transfer', choices=['zeroshot', 'transfer', 'fine-tuning'])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
