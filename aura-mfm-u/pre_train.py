import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import torch
from lib.utils import bool_flag

class Logger:
    def __init__(self, log_dir, debug=False):
        if debug:
            self.file = sys.stdout
        else:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = os.path.join(log_dir, f"{len(glob.glob(f'{log_dir}/*')):03}")
            self.ckpt_dir = os.path.join(self.log_dir, 'ckpts')
            os.makedirs(self.ckpt_dir, exist_ok=True)        
            self.file = open(os.path.join(self.log_dir, 'out.log'), 'w')
    def write(self, msg):
        self.file.write(msg)
    def flush(self):
        self.file.flush()

#######
# main
#######
def main(args):
    ##############
    # Set up meta
    ##############
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger = Logger(args.log_dir, debug=args.debug)
    sys.stdout = logger
    
    print('\n## Configuration: ')
    for k, v in vars(args).items():
        print(k, v)
    
    
    ##############
    # Set up data
    ##############
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from lib.dataloader import CrossModalDataset, worker_init_fn
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    
    train_dataset = CrossModalDataset(path=args.data_path, modality=args.modality, name='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn, shuffle=True, pin_memory=True)

    val_dataset = CrossModalDataset(path=args.data_path, modality=args.modality, name='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn, shuffle=False, pin_memory=True)

    ###############
    # Set up model
    ###############
    from lib.models import MW2StackRNNPooling, MW2StackRNNPoolingMocap, CrossModalContrastiveLearningModule
    from lib.clip_model import ClipPLModel
    import lib.transformer as Ts
    encoders = {}
    if 'imu' in args.modality:
        imu_encoder = (
            MW2StackRNNPooling(size_embeddings=args.size_embedding)                                       if args.arch == 'rnn' else
            Ts.__dict__[args.model_size](window_size=1000, in_chans=6, patch_size=args.patch_size['imu']) if args.arch == 'transformer' else None
        )
        encoders['imu'] = imu_encoder
    if 'mocap' in args.modality:
        mocap_encoder = (
            MW2StackRNNPoolingMocap(size_embeddings=args.size_embedding)                                   if args.arch == 'rnn' else
            Ts.__dict__[args.model_size](window_size=50, in_chans=51, patch_size=args.patch_size['mocap']) if args.arch == 'transformer' else None
        )
        encoders['mocap'] = mocap_encoder
    if 'text' in args.modality:
        text_encoder = ClipPLModel(freeze=True, device=device).to(dtype=torch.float)
        encoders['text'] = text_encoder
    
    model = CrossModalContrastiveLearningModule(encoders, args.modality, device=device)

    ##################
    # Set up training
    ##################
    from lib.evaluation import evaluate, evaluate_model
    from lib.loss import CalcLoss
    loss_fn = CalcLoss(args.modality)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)

    best_mrr = 1e-9
    log_interval = 1
    train_losses, val_losses = [], []

    print('\n\nStart training')

    model.to(device)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, xs in enumerate(train_loader):
            out = model(xs)
            loss = loss_fn(out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_interval == 0:
                print(f'Epoch:[{epoch}/{args.epochs}] Step:[{i}/{len(train_loader)}] Loss:{loss.item():.6f}')
        
        average_loss = running_loss/len(train_loader)
        val_loss = evaluate_model(model, val_loader, loss_fn=loss_fn, device=device)
        train_losses.append(average_loss)
        val_losses.append(val_loss)

        target = 'mocap' if 'mocap' in args.modality else 'text'
        metrics = evaluate(val_loader, model, source='imu', target=target, device=device)
        print(f'# Epoch:[{epoch}/{args.epochs}] Loss:[train:{average_loss:.6f} val:{val_loss:.6f}]')
        print(f'# Retrieval: s -> t:{metrics["s_t_metrics"]}')
        print(f'# Retrieval: T -> s:{metrics["t_s_metrics"]}')

        if args.debug:
            continue

        val_mrr = metrics['s_t_metrics']['MRR'] + metrics['t_s_metrics']['MRR']
        if metrics['s_t_metrics']['MRR']+metrics['t_s_metrics']['MRR'] > best_mrr and epoch > 100:
            best_mrr = val_mrr
            best_epoch = epoch
            best_ckpt = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'args': args}
        if epoch % (args.epochs//10) == 0 and epoch != 0:
            ckpt = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'args': args}
            torch.save(ckpt, os.path.join(logger.ckpt_dir, f'{epoch:05}.pt'))
    
    torch.save(best_ckpt, os.path.join(logger.log_dir, f'best_{best_epoch}.pt'))

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig(os.path.join(logger.log_dir, 'loss.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--data-path", type=str,  default='/home/ukita/data4/dataSpace/EGO_EXO4D/preprocessed/datasets/i-t_datafiles')
    parser.add_argument("--log-dir", type=str, default='./experiment_log/pre_train')
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--modality", type=list, default=['imu', 'text'])         # ['imu', 'mocap'] or ['imu', 'text']
    parser.add_argument("--arch", type=str, default='rnn', choices=['rnn', 'transformer'])
    parser.add_argument("--size-embedding", type=int, default=512)                 # --archがrnnの時表現の次元数
    parser.add_argument("--model-size", type=str, default='SS')                    # --archがtransformerの時モデルのサイズ
    parser.add_argument("--patch-size", type=dict, default={'imu': 8, 'mocap': 1}) # --archがtransformerの時各エンコーダのパッチサイズ
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    args = parser.parse_args()

    main(args)