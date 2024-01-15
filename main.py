import torch
import torch.nn as nn
from myhubert_dataset import myAVHubertDataset
import os
from os.path import join
import pdb
from hubert_avse import AVSEHubertModel
import time
from torch.utils.tensorboard import SummaryWriter
import datetime
import myutils
import tqdm

# steps = 0

def train_one_epoch(model, data_loader, device, criterion, optimizer, epoch, writer):
    global steps
    model.train()
    train_loss = 0.
    for video_feats, noisy_feats, clean_spec, noisy_spec, _, _, _ in tqdm.tqdm(data_loader):
        video_feats, noisy_feats, clean_spec, noisy_spec = \
            video_feats.to(device), noisy_feats.to(device), clean_spec.to(device), noisy_spec.to(device)
        out = model(video_feats, noisy_feats, noisy_spec)
        loss = criterion(out,clean_spec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if steps % 20 == 0:
            # writer.add_scalar('Loss/train', round(loss.item(),7), steps)
        # steps += 1

        train_loss += loss.item()

    train_loss /= len(data_loader)
    train_loss = round(train_loss, 7)
    print(f"epoch: {epoch}, training loss: {train_loss}")
    writer.add_scalar('Loss/train', train_loss, epoch)


def evaluate(model, data_loader, device, criterion, epoch, writer):
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for video_feats, noisy_feats, clean_spec, noisy_spec, _, _, _ in tqdm.tqdm(data_loader):
            video_feats, noisy_feats, clean_spec, noisy_spec = \
                video_feats.to(device), noisy_feats.to(device), clean_spec.to(device), noisy_spec.to(device)

            out = model(video_feats, noisy_feats, noisy_spec)
            loss = criterion(out,clean_spec)
            val_loss += loss.item()

    val_loss /= len(data_loader)
    val_loss = round(val_loss, 7)
    print(f"\t\tval loss: {val_loss}")
    if writer:
        writer.add_scalar('Loss/validation', val_loss, epoch)

    return val_loss




def main(args):

    myutils.init_distributed_mode(args)

    args.output_dir = myutils.save_path_formatter(args)
    print(args)
    writer = SummaryWriter(args.output_dir,flush_secs=30)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    
    torch.backends.cudnn.benchmark = True

    print("Loading data")
    dataset = myAVHubertDataset(join(args.data_path,args.train_list), no_ssl=args.no_ssl)
    dataset_test = myAVHubertDataset(join(args.data_path,args.val_list), no_ssl=args.no_ssl)
    
    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=None, num_workers=args.workers, shuffle=True,
        pin_memory=True, collate_fn=dataset.collater)    

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=None, num_workers=args.workers, shuffle=False,
        pin_memory=True, collate_fn=dataset.collater)    

    print("Creating model")
    model = AVSEHubertModel(args.ckpt_path, mapping=args.mapping, from_scratch=args.from_scratch, no_ssl=args.no_ssl, no_video=args.no_video, crossfeat=args.crossfeat)
    model.to(device)
    if args.freeze_extractor:
        for name, param in model.model.named_parameters():
            if "extract" in name:
                param.requires_grad = False
    if args.no_finetune:
        for name, param in model.model.named_parameters():
            param.requires_grad = False


    lr = args.lr
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': model.model.parameters()},
            {'params': model.blstm.parameters(), 'lr': lr*0.1}
            ], lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam': 
        optimizer = torch.optim.Adam([
            {'params': model.model.parameters()},
            {'params': model.blstm.parameters(), 'lr': lr*0.1}
            ], lr=lr, weight_decay=args.weight_decay) 
    elif args.optimizer == 'AdamW': 
        optimizer = torch.optim.AdamW([
            # {'params': model.model.parameters()},
            {'params': model.blstm.parameters(), 'lr': lr*0.1}
            ], lr=lr, weight_decay=args.weight_decay) 
    else:
        raise ValueError('Please assign an optimizer.') 

    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError('Please assign a loss.') 
    

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device, criterion, args.start_epoch, writer=None)
        return

    print("Start training")
    start_time = time.time()
    best_val_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, data_loader, device, criterion, optimizer, epoch, writer)
        cur_val_loss = evaluate(model, data_loader_test, device, criterion, epoch, writer)
        if not args.not_save_model:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args}
            myutils.save_on_master(
                checkpoint,
                join(args.output_dir, 'model_{}.pth'.format(epoch)))


            if cur_val_loss < best_val_loss:
                myutils.save_on_master(
                    checkpoint,
                    join(args.output_dir, 'bestmodel.pth'))

                best_val_loss = cur_val_loss


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    writer.close()


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='TMHINT', help='dataset')
    parser.add_argument('--data-path', default='../data', help='data path')
    parser.add_argument('--train-list', default='train.lst', help='name of train list')
    parser.add_argument('--val-list', default='val.lst', help='name of validation list')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')    
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--optimizer', default='Adam', help='optimizer')
    parser.add_argument('--loss', default='mse', help='loss type')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--ckpt-path', default='/home/richardleelai/jchou/project/avhubert_avse/code/checkpoint/base_lrs3_iter5.pt', help='pretrained checkpoint')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch') 
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    ) 
    parser.add_argument(
        "--freeze-extractor",
        dest="freeze_extractor",
        action="store_true",
    ) 
    parser.add_argument(
        "--no-finetune",
        dest="no_finetune",
        action="store_true",
    ) 
    parser.add_argument(
        "--from-scratch",
        dest="from_scratch",
        action="store_true",
    )     
    parser.add_argument(
        "--not-save-model",
        dest="not_save_model",
        help="not save model",
        action="store_true",
    ) 
    parser.add_argument(
        "--no-ssl",
        dest="no_ssl",
        help="not use ssl model",
        action="store_true",
    )
    parser.add_argument(
        "--no-video",
        dest="no_video",
        help="not use video",
        action="store_true",
    )
    parser.add_argument(
        "--mapping",
        dest="mapping",
        help="use mapping not masking",
        action="store_true",
    ) 
    parser.add_argument(
        "--crossfeat",
        dest="crossfeat",
        help="add cross domain feature",
        action="store_true",
    ) 
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()
    main(args)

