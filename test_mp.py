import torch
import torch.nn as nn
from myhubert_dataset import myAVHubertDataset
import os
from os.path import join
import pdb
from hubert_avse import AVSEHubertModel
import time
import datetime
import myutils
import tqdm
import sys
import torchaudio
import pandas as pd
from multiprocessing import Pool
from itertools import cycle
import torch.nn.functional as F
import numpy as np

def run(data):

    video_fn, clean_fn, noisy_fn, args = data


    device = torch.device("cpu")
    dataset = myAVHubertDataset(join(args.data_path,args.test_list))

    model = AVSEHubertModel(args.ckpt_path)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])


    video_feats, _, noisy_feats, _, noisy_spec, noisy_phase, length, mix_name = dataset.load_feature((video_fn, clean_fn, noisy_fn))

    noisy_feats, video_feats = torch.from_numpy(noisy_feats.astype(np.float32)), torch.from_numpy(video_feats.astype(np.float32))

    if True:
    # if self.normalize:
        with torch.no_grad():
            noisy_feats = F.layer_norm(noisy_feats, noisy_feats.shape[1:])

    model.eval()

    with torch.no_grad():
        video_feats, noisy_feats, noisy_spec, noisy_phase = \
            video_feats.permute(3,0,1,2).unsqueeze(0).to(device), noisy_feats.permute(1,0).unsqueeze(0).to(device), \
            noisy_spec.permute(1,0).unsqueeze(0).to(device), noisy_phase.permute(1,0,2).unsqueeze(0).to(device)

        out = model(video_feats, noisy_feats, noisy_spec)

        enhan_wav = myutils.feature_to_wav((out.permute(0,2,1), noisy_phase.permute(0,2,1,3)), length)
        root = mix_name[0]
        clean_wav,sr = torchaudio.load(join(root,clean_fn))
        myutils.mkdir(os.path.dirname(join(args.output_dir,noisy_fn)))
        torchaudio.save(join(args.output_dir,noisy_fn), enhan_wav, sr)
        noisy_wav = myutils.feature_to_wav((noisy_spec.permute(0,2,1), noisy_phase.permute(0,2,1,3)), length)

        s_pesq, s_stoi, s_snr, s_sdr = myutils.cal_score(clean_wav.squeeze().detach().numpy(),enhan_wav.squeeze().detach().numpy())
        with open(args.enhan_score_path, 'a') as f:
            f.write(f'{noisy_fn},{s_pesq},{s_stoi},{s_snr},{s_sdr}\n')
        s_pesq, s_stoi, s_snr, s_sdr = myutils.cal_score(clean_wav.squeeze().detach().numpy(),noisy_wav.squeeze().detach().numpy())
        with open(args.noisy_score_path, 'a') as f:
            f.write(f'{noisy_fn},{s_pesq},{s_stoi},{s_snr},{s_sdr}\n')


def main(args):

    myutils.init_distributed_mode(args)
    args.output_dir = 'C:/Users/batma/Documents/avhubert/output'
    myutils.mkdir(args.output_dir)
    print(args)

    device = torch.device("cpu")
    args.dataset_test = myAVHubertDataset(join(args.data_path,args.test_list))

    args.enhan_score_path = join(args.output_dir,"score_enhanced.csv")
    args.noisy_score_path = join(args.output_dir,"score_noisy.csv")

    with open(args.enhan_score_path, 'a') as f:
        f.write('Filename,PESQ,STOI,SISNR,SDR\n')

    with open(args.noisy_score_path, 'a') as f:
        f.write('Filename,PESQ,STOI,SISNR,SDR\n')


    with open(join(args.data_path,args.test_list)) as f:
        root = f.readline()
        lines = f.readlines()

    import re
    video_fn_lst, clean_fn_lst, noisy_fn_lst = [], [], []
    for line in lines:
        tmp = re.split('\t|\n',line)
        video_fn_lst.append(tmp[0])
        clean_fn_lst.append(tmp[1])
        noisy_fn_lst.append(tmp[2])


    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm.tqdm(pool.imap_unordered(run, zip(video_fn_lst, clean_fn_lst, noisy_fn_lst, args_list))):
        None  

    data = pd.read_csv(args.enhan_score_path)
    pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
    stoi_mean = data['STOI'].to_numpy().astype('float').mean()
    snr_mean  = data['SISNR'].to_numpy().astype('float').mean()
    sdr_mean  = data['SDR'].to_numpy().astype('float').mean()
    with open(args.enhan_score_path, 'a') as f:
        f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(snr_mean),str(sdr_mean)))+'\n')


    data = pd.read_csv(args.noisy_score_path)
    pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
    stoi_mean = data['STOI'].to_numpy().astype('float').mean()
    snr_mean  = data['SISNR'].to_numpy().astype('float').mean()
    sdr_mean  = data['SDR'].to_numpy().astype('float').mean()
    with open(args.noisy_score_path, 'a') as f:
        f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(snr_mean),str(sdr_mean)))+'\n')


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='TMHINT', help='dataset')
    parser.add_argument('--data-path', default='../data', help='data path')
    parser.add_argument('--root', default='../data', help='data path')
    parser.add_argument('--test-list', default='C:/Users/batma/Documents/avhubert/test.lst', help='name of test list')
    parser.add_argument('--ckpt-path', default='C:/Users/batma/Documents/avhubert/base_lrs3_iter5.pt', help='pretrained checkpoint')
    parser.add_argument('--resume', default='', help='checkpoint')
    parser.add_argument('--output-dir', default='C:/Users/batma/Documents/avhubert/output', help='path where to save')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()
    main(args)

