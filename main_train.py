# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import os

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from datasets import SeismicDataset
from mask_generator import TubeMaskingGenerator
import model_mae as mae 



def get_args_parser():
    parser = argparse.ArgumentParser('3D MAE training', add_help=False)
    parser.add_argument('--batch_size', default=5, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=40000, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--frames', default=7, type=float,
                        help='no. of frames to be sampled from volume')
    
    parser.add_argument('--frame_distance', default=5, type=float,
                        help='inter frame distance while sampling from volume.')

    parser.add_argument('--patch_size', default=16, type=int,
                        help='patch size for 3d convolution')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='dataset', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    return parser

def main(args):
    device = torch.device(args.device)

    cudnn.benchmark = True
    dataset_train =  SeismicDataset(args.data_path, args.frames, args.frame_distance)   
    print ("dataset len", len(dataset_train))
    h_p = args.input_size//args.patch_size
    maskingGen = TubeMaskingGenerator((args.frames, h_p, h_p), args.mask_ratio)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=True,
    )

    model = mae.mae_vit_mini_patch16()
    model = model.to(device)

    no_decay_param = model.no_weight_decay()
    decay_param_list = []
    no_decay_param_list = []
    for name, m in model.named_parameters():
    
        if name in no_decay_param or "norm" in name:
            no_decay_param_list.append(m)
        else:
            decay_param_list.append(m)

    optimizer=torch.optim.AdamW([{'params': no_decay_param_list, 'weight_decay': 0}, {'params': decay_param_list}], lr=args.lr)
    scaler=torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,steps_per_epoch=len(dataset_train)//args.batch_size, epochs=args.epochs,pct_start=0.1,)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    tb_iter = 0
    model.train()
    for epoch in range(args.epochs):
        losss=0
        time=tqdm(range(len(data_loader_train)))
        for i,(seismic_data) in enumerate(data_loader_train):
            tb_iter = tb_iter + 1
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                x = seismic_data.unsqueeze(1).to(torch.float32).to(device)
                mask = np.vstack(maskingGen() for _ in range(x.shape[0]))
                loss, pred, mask = model(x, mask) 
    
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            
        
            losss=(losss*i+loss.item())/(i+1)   
            if log_writer is not None:
                log_writer.add_scalar('Loss/train', losss, tb_iter)
                log_writer.add_scalar('lr/train', scheduler.get_last_lr()[0], tb_iter)
    
            time.set_description(f"train_epoch:{epoch},loss:{losss}, lr:{optimizer.param_groups[0]['lr']:.4e} ")
            time.update()
        time.close()
    
        if (epoch+ 1) %100 == 0:
            filename = f"./mae_3d_{epoch}_loss{losss}.pt"
            torch.save(model.state_dict(), os.path.join(args.log_dir, filename))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
