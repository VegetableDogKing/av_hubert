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

def evaluate(model, data_loader, device, enhan_score_path, noisy_score_path):

    model.eval()

    with torch.no_grad():
        for video_feats, noisy_feats, _, noisy_spec, noisy_phase, length, filename in tqdm.tqdm(data_loader):
            video_feats, noisy_feats, noisy_spec, noisy_phase = \
                video_feats.to(device), noisy_feats.to(device), noisy_spec.to(device), noisy_phase.to(device)
            
            root, video_fn, clean_fn, noisy_fn = filename[0]

            out = model(video_feats, noisy_feats, noisy_spec)
            
            enhan_wav = myutils.feature_to_wav((out.permute(0,2,1), noisy_phase.permute(0,2,1,3)), length[0])
            clean_wav,sr = torchaudio.load(join(root,clean_fn))
            myutils.mkdir(os.path.dirname(join(args.output_dir,noisy_fn)))
            torchaudio.save(join(args.output_dir,noisy_fn), enhan_wav, sr)
            noisy_wav = myutils.feature_to_wav((noisy_spec.permute(0,2,1), noisy_phase.permute(0,2,1,3)), length[0])

            s_pesq, s_stoi, s_snr, s_sdr = myutils.cal_score(clean_wav.squeeze().detach().numpy(),enhan_wav.squeeze().detach().numpy())
            print('s_pesq: ', s_pesq)
            print('s_stoi: ', s_stoi)
            with open(enhan_score_path, 'a') as f:
                f.write(f'{noisy_fn},{s_pesq},{s_stoi},{s_snr},{s_sdr}\n')
            s_pesq, s_stoi, s_snr, s_sdr = myutils.cal_score(clean_wav.squeeze().detach().numpy(),noisy_wav.squeeze().detach().numpy())
            with open(noisy_score_path, 'a') as f:
                f.write(f'{noisy_fn},{s_pesq},{s_stoi},{s_snr},{s_sdr}\n')



def main(args):

    myutils.init_distributed_mode(args)
    args.output_dir = 'C:/Users/batma/Documents/avhubert/output'
    #args.output_dir = os.path.dirname(args.resume.replace("logs", "enhan_result_jasa"))
    myutils.mkdir(args.output_dir)
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

    enhan_score_path = join(args.output_dir,"score_enhanced.csv")
    noisy_score_path = join(args.output_dir,"score_noisy.csv")

    with open(enhan_score_path, 'a') as f:
        f.write('Filename,PESQ,STOI,SISNR,SDR\n')

    with open(noisy_score_path, 'a') as f:
        f.write('Filename,PESQ,STOI,SISNR,SDR\n')

    evaluate(model, data_loader, device, enhan_score_path, noisy_score_path)

    data = pd.read_csv(enhan_score_path)
    
    pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
    stoi_mean = data['STOI'].to_numpy().astype('float').mean()
    snr_mean  = data['SISNR'].to_numpy().astype('float').mean()
    sdr_mean  = data['SDR'].to_numpy().astype('float').mean()
    with open(enhan_score_path, 'a') as f:
        f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(snr_mean),str(sdr_mean)))+'\n')


    data = pd.read_csv(noisy_score_path) 
    pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
    stoi_mean = data['STOI'].to_numpy().astype('float').mean()
    snr_mean  = data['SISNR'].to_numpy().astype('float').mean()
    sdr_mean  = data['SDR'].to_numpy().astype('float').mean()
    with open(noisy_score_path, 'a') as f:
        f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(snr_mean),str(sdr_mean)))+'\n')



def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='TMHINT', help='dataset')
    parser.add_argument('--data-path', default='', help='data path')
    parser.add_argument('--root', default='', help='data path')
    parser.add_argument('--test-list', default='test.lst', help='name of test list')
    parser.add_argument('--ckpt-path', default='base_lrs3_iter5.pt', help='pretrained checkpoint')
    parser.add_argument('--resume', default='', help='checkpoint')
    parser.add_argument('--output-dir', default='C:/Users/batma/Documents/avhubert/output', help='path where to save')

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

