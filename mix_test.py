from preprocess import get_filepaths, check_path, _gen_noisy
import pdb
from tqdm import tqdm
from scipy.io.wavfile import read as audioread
from scipy.io.wavfile import write as audiowrite
import os
import librosa

if __name__ == '__main__':
    test_list = "/home/richardleelai/jchou/project/avhubert_avse/data/test3.lst"
    names = []
    with open(test_list) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")

            clean_path = items[1]
            noisy_path = items[2]
            video_path = items[0]
            names.append((noisy_path, clean_path))

    for (t, r) in tqdm(names):

        sp = t.split('/')[0]
        n_type = t.split('/')[4].split("_")[0]
        snr = t.split('/')[4].split("_")[-1][:-2]
        if snr == 'n7':
            snr = -7
        elif snr == 'n4':
            snr = -4
        elif snr == 'n1':
            snr = -1            
        else:
            snr = int(snr)


        if n_type == 'babble':
            n_file = os.path.join("noises", "babble.wav")
            
        elif n_type == 'babycry':
            n_file = os.path.join("noises",  "babycry.wav")            
        
        elif n_type == '1talker':
            n_file = os.path.join("noises", "1talker_vctk.wav") 
             
        elif n_type == '2talker':
            n_file = os.path.join("noises", "M_2talker_tmhint.wav")  
            
        elif n_type == '3talker':
            n_file = os.path.join("noises", "M_3talkers_dns_challenge.wav")              


        # sr ,y_noise = audioread(n_file)
        # y_noise = y_noise.astype('float32')
        y_noise,sr = librosa.load(n_file,sr=16000)
        # pdb.set_trace()

        c_file = os.path.join(root, r)

        c_name = c_file.split('/')[-1]
        # sr ,y_clean = audioread(c_file)
        # y_clean = y_clean.astype('float32')
        y_clean,sr = librosa.load(c_file,sr=16000)

        save_path = '/home/richardleelai/Data/user_jchou/AVSE/data/TMHINT/{}/audio/noisy_enh/test/{}_n{}db/{}'.format(sp, n_type, -1*snr, c_name) if snr < 0 else \
                    '/home/richardleelai/Data/user_jchou/AVSE/data/TMHINT/{}/audio/noisy_enh/test/{}_{}db/{}'.format(sp, n_type, snr, c_name)
        check_path(os.path.dirname(save_path))

        # if os.path.exists(save_path):
        #     continue

        _gen_noisy(y_clean,y_noise,snr,save_path, sr)

