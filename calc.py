# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:46:05 2023

@author: user
"""

from pesq import pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources
import math
import pandas
import os

def cal_score(clean,enhanced):
#     if not clean.shape==enhanced.shape:
#         enhanced = enhanced[:clean.shape]
#     clean = clean/abs(clean).max()
#     enhanced = enhanced/abs(enhanced).max()
    try:
        s_stoi = stoi(clean, enhanced, 16000)
    except:
        s_stoi = 0 
#     s_pesq = pesq(clean, enhanced, 16000)
    s_pesq = pesq(16000, clean, enhanced, 'nb')

    if math.isnan(s_pesq):
        s_pesq=0
    if math.isnan(s_stoi):
        s_stoi=0
    return round(s_pesq,5), round(s_stoi,5)

clean = '/home/richardleelai/Data/user_jchou/AVSE/data/TMHINT/SP05/audio/noisy_enh/test/'
enhanced = '/home/richardleelai/jchou/project/avhubert_avse/code/logmmse/'

for i in range(len(next(os.walk(clean))[1])):
    results = cal_score(clean, enhanced)
    print(results)
    
