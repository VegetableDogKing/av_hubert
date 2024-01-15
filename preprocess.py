from glob import iglob
import librosa
from multiprocessing import Pool
from functools import partial
import pdb
import os
from random import randint
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read as audioread
from scipy.io.wavfile import write as audiowrite
from shutil import copyfile
import random
import soundfile

random.seed( 10 )

def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)
    
def get_filepaths(directory,ftype='wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)
    
def _gen_noisy(y_clean,y_noise,SNR,save_path, sr):

    clean_pwr = sum(y_clean**2) / len(y_clean)
    
    if len(y_noise) < len(y_clean):
        tmp = (len(y_clean) // len(y_noise)) + 1
        y_noise = np.array([x for j in [y_noise] * tmp for x in j])

    start = randint(0,len(y_noise)-len(y_clean))
    y_noise = y_noise[start:start+len(y_clean)]
#     y_noise = y_noise - np.mean(y_noise)
    noise_pwr = sum(y_noise**2) / len(y_noise)
    noise_variance = (clean_pwr/noise_pwr) / (10**(SNR / 10))
    noise = np.sqrt(noise_variance) * y_noise

    y_noisy = np.clip((y_clean + noise), -1., 1.)

    assert (abs(y_noisy).max())<=1., f"file: {save_path}, abs(y_noisy).max(): {abs(y_noisy).max()}"
    # y_noisy, y_clean = normalize(y_noisy),normalize(y_clean)
    # audiowrite(save_path,sr,y_noisy/5)
    soundfile.write(save_path, y_noisy, sr)

    
def normalize(noisy,clean=np.array([0])):
    maxv = np.iinfo(np.int16).max
    time = maxv/(abs(noisy).max())
    noisy = noisy*time
#     pdb.set_trace()
    if clean.any():
        clean = clean*time
        return noisy.astype('int16'),clean.astype('int16')
    return noisy.astype('int16')
    
def _gen_noisy_org(y_clean,y_noise,SNR,save_path, sr):

    clean_pwr = sum(y_clean**2) / len(y_clean)
    
    if len(y_noise) < len(y_clean):
        tmp = (len(y_clean) // len(y_noise)) + 1
        y_noise = np.array([x for j in [y_noise] * tmp for x in j])

    start = randint(0,len(y_noise)-len(y_clean))
    y_noise = y_noise[start:start+len(y_clean)]
#     y_noise = y_noise - np.mean(y_noise)
    noise_pwr = sum(y_noise**2) / len(y_noise)
    noise_variance = (clean_pwr/noise_pwr) / (10**(SNR / 10))
    noise = np.sqrt(noise_variance) * y_noise

    y_noisy = y_clean + noise

    
    y_noisy, y_clean = normalize(y_noisy),normalize(y_clean)
    audiowrite(save_path,sr,y_noisy)
    

    #### scipy cannot conver TIMIT format ####
modes=['Train','Test']
c_path = '../TMHINT/OriginalSound'
folders = {'Train':['b1','b2','b3','g1','g2','g3'],
           'Test':['b4','g4']
          }
# for mode in modes:
#     for folder in folders[mode]:
#         wav_files = get_filepaths(os.path.join(c_path,folder))
#         for wav_file in tqdm(wav_files):
#             filename = wav_file.split('/')[-1]
#             person = wav_file.split('/')[-2]
#             n_name = person+'_'+filename
#             out_path = '../data/{}/Clean'.format(mode)
#             copyfile(wav_file,os.path.join(out_path,n_name))
        
    
    
# snrs = [-10,-7,-4,-1,0,1,4,7,10]
snrs = [2,5]
# n_path = './noise'

modes=['Test']

# for mode in modes:
    
#     n_path = '../data_n/{}/Noise'.format(mode)
#     n_files = get_filepaths(n_path)
#     c_path = '../data/{}/Clean'.format(mode)
#     c_files = get_filepaths(c_path)
#     for c_file in tqdm(c_files):
#         c_name = c_file.split('/')[-1]
#         sr ,y_clean = audioread(c_file)
#         y_clean = y_clean.astype('float64')
#         for snr in snrs:
#             n_file = random.choice(n_files)
#             n_type = n_file.split('/')[-1].split('.')[0]
#             y_noise,sr = librosa.load(n_file,sr=16000)
#             save_path = '../data_n/{}/Noisy/{}/{}/{}'.format(mode,n_type,snr,c_name)
#             check_folder(save_path)
#             _gen_noisy(y_clean,y_noise,snr,save_path)
            
for mode in modes: 
    n_path = '../data_n/{}/Noise'.format(mode)
    n_files = get_filepaths(n_path)
    for n_file in n_files:
#         pdb.set_trace()
        try:
            n_type = n_file.split('/')[-1].split('.')[0]
            y_noise,sr = librosa.load(n_file,sr=16000)
#             sr ,y_noise = audioread(n_file)
#             y_noise = y_noise.astype('float64')
        except:
            pdb.set_trace()
        
        c_path = '../data/{}/Clean'.format(mode)
        c_files = get_filepaths(c_path)
        
        for c_file in tqdm(c_files):
            c_name = c_file.split('/')[-1]
            sr ,y_clean = audioread(c_file)
            y_clean = y_clean.astype('float64')
            
            for snr in snrs:
                save_path = '../data_n/{}/Noisy/{}/{}/{}'.format(mode,n_type,snr,c_name)
                check_folder(save_path)
                _gen_noisy(y_clean,y_noise,snr,save_path)
                