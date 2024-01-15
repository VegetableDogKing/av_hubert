import myutils
from scipy.io import wavfile
import pdb
clean_lst = "./baseline_result/clean.txt"
baseline_lst = "./baseline_result/klt.txt"
enhan_score_path = "./baseline_result/klt/score_enhanced.csv"

with open(enhan_score_path, 'a') as f:
    f.write('Filename,PESQ,STOI,SISNR,SDR\n')

with open(clean_lst) as file:
    lines = file.readlines()
    clines = [line.rstrip() for line in lines]

with open(baseline_lst) as file:
    lines = file.readlines()
    blines = [line.rstrip() for line in lines]

from tqdm.contrib import tzip

for (c, b) in tzip(clines, blines):

    sr, clean_wav = wavfile.read(c)
    sr, enhan_wav = wavfile.read(b)


    if len(clean_wav)>=len(enhan_wav):
        clean_wav = clean_wav[:len(enhan_wav)]
    elif len(clean_wav[0])<len(enhan_wav):
        enhan_wav = enhan_wav[:len(clean_wav)]

    s_pesq, s_stoi, s_snr, s_sdr = myutils.cal_score(clean_wav,enhan_wav)

    with open(enhan_score_path, 'a') as f:
        f.write(f'{b},{s_pesq},{s_stoi},{s_snr},{s_sdr}\n')

import pandas as pd

data = pd.read_csv(enhan_score_path)
pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
stoi_mean = data['STOI'].to_numpy().astype('float').mean()
snr_mean  = data['SISNR'].to_numpy().astype('float').mean()
sdr_mean  = data['SDR'].to_numpy().astype('float').mean()
with open(enhan_score_path, 'a') as f:
    f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(snr_mean),str(sdr_mean)))+'\n')
