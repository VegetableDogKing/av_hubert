import torch
import torch.nn as nn
from myhubert_dataset import myAVHubertDataset
import os
from os.path import join
import pdb
from hubert_avse_layer_embedding import AVSEHubertModel
import time
import datetime
import myutils
import tqdm
import sys
import torchaudio
import pandas as pd

def evaluate(model, data_loader, device):

    model.eval()

    with torch.no_grad():
        for video_feats, noisy_feats, _, noisy_spec, noisy_phase, length, filename in tqdm.tqdm(data_loader):
            video_feats, noisy_feats, noisy_spec, noisy_phase = \
                video_feats.to(device), noisy_feats.to(device), noisy_spec.to(device), noisy_phase.to(device)
            
            root, video_fn, clean_fn, noisy_fn = filename[0]

            out, layer_embedding = model(video_feats, noisy_feats, noisy_spec)
            
            pdb.set_trace()



def main(args):

    myutils.init_distributed_mode(args)
    #args.output_dir = os.path.dirname(args.resume.replace("logs", "enhan_result"))
    #args.output_dir = os.path.dirname(args.resume.replace("logs", "enhan_result_jasa"))
    #33myutils.mkdir(args.output_dir)
    print(args)

    device = torch.device("cpu")


    print("Loading data")
    dataset = myAVHubertDataset(join(args.data_path,args.test_list), no_ssl=args.no_ssl)
    
    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=None, num_workers=0, shuffle=False,
        pin_memory=False, collate_fn=dataset.collater)    

    print("Creating model")
    model = AVSEHubertModel(args.ckpt_path, no_ssl=args.no_ssl, no_video=args.no_video, crossfeat=args.crossfeat)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])



    evaluate(model, data_loader, device)





def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='TMHINT', help='dataset')
    parser.add_argument('--data-path', default='../data', help='data path')
    parser.add_argument('--root', default='../data', help='data path')
    parser.add_argument('--test-list', default='test.lst', help='name of test list')
    parser.add_argument('--ckpt-path', default='/home/richardleelai/jchou/project/avhubert_avse/code/checkpoint/base_lrs3_iter5.pt', help='pretrained checkpoint')
    parser.add_argument('--resume', default='', help='checkpoint')
    parser.add_argument('--output-dir', default='', help='path where to save')

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

