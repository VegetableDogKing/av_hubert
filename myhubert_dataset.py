# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
sys.path.append('./av_hubert/fairseq')
# from fairseq.data import data_utils
# from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile
import pdb
sys.path.append('./av_hubert/avhubert')
import utils as custom_utils



def load_audio_visual(manifest_path):

    names = []

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")

            clean_path = items[1]
            noisy_path = items[2]
            video_path = items[0]
            names.append((video_path, clean_path, noisy_path))


    return root, names

class myAVHubertDataset():
    def __init__(
            self,
            manifest_path: str,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            normalize: bool = True,
            no_ssl = False,
    ):

        self.image_crop_size = image_crop_size
        self.root, self.names = load_audio_visual(manifest_path)
        self.transform = custom_utils.Compose([
            custom_utils.Normalize( 0.0,255.0 ),
            custom_utils.CenterCrop((image_crop_size, image_crop_size)),
            custom_utils.Normalize(image_mean, image_std) ])
        self.stack_order_audio = 4
        self.F = self.stack_order_audio * 26
        self.normalize = normalize

        self.n_fft   = 512
        self.win_length = 400
        self.hop_length = 160
        self.epsilon  = 1e-6

        self.no_ssl = no_ssl

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats
        

    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, clean_fn, noisy_fn = mix_name
        video_feats = self.load_video(video_fn) # [T, H, W, 1]
        sample_rate_clean, wav_data_clean = wavfile.read(os.path.join(self.root, clean_fn))
        sample_rate_noisy, wav_data_noisy = wavfile.read(os.path.join(self.root, noisy_fn))
        assert sample_rate_clean == 16_000 and len(wav_data_clean.shape) == 1
        assert sample_rate_noisy == 16_000 and len(wav_data_noisy.shape) == 1
        clean_feats = logfbank(wav_data_clean, samplerate=sample_rate_clean).astype(np.float32) # [T, F]
        clean_feats = stacker(clean_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
        noisy_feats = logfbank(wav_data_noisy, samplerate=sample_rate_noisy).astype(np.float32) # [T, F]
        noisy_feats = stacker(noisy_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]


        # extract log spectorgram
        clean_log_spec, _, _ = self.get_spectrogram(os.path.join(self.root, clean_fn))
        noisy_log_spec, noisy_phase, length = self.get_spectrogram(os.path.join(self.root, noisy_fn))

        if not self.no_ssl:
            if clean_feats is not None and noisy_feats is not None and video_feats is not None:
                assert len(clean_feats)==len(noisy_feats)
                diff = len(clean_feats) - len(video_feats)
                if diff < 0:
                    clean_feats = np.concatenate([clean_feats, np.zeros([-diff, clean_feats.shape[-1]], dtype=clean_feats.dtype)])
                    noisy_feats = np.concatenate([noisy_feats, np.zeros([-diff, noisy_feats.shape[-1]], dtype=noisy_feats.dtype)])
                elif diff > 0:
                    clean_feats = clean_feats[:-diff]
                    noisy_feats = noisy_feats[:-diff]


            clean_feats, noisy_feats, video_feats = torch.from_numpy(clean_feats.astype(np.float32)), \
                torch.from_numpy(noisy_feats.astype(np.float32)), torch.from_numpy(video_feats.astype(np.float32)) 
                                   
            return video_feats, clean_feats, noisy_feats, clean_log_spec.permute(1,0), \
                    noisy_log_spec.permute(1,0), noisy_phase.permute(1,0,2), length, (self.root,)+mix_name 

        else:
            clean_log_spec = clean_log_spec.permute(1,0)
            noisy_log_spec = noisy_log_spec.permute(1,0)
            noisy_phase = noisy_phase.permute(1,0,2)

            clean_feats, noisy_feats, video_feats = torch.from_numpy(clean_feats.astype(np.float32)), \
                torch.from_numpy(noisy_feats.astype(np.float32)), torch.from_numpy(video_feats.astype(np.float32)) 

            if video_feats is not None:
                diff = len(noisy_log_spec) - len(video_feats)
                if diff>0:
                    video_feats = torch.cat((video_feats, torch.zeros((1,self.image_crop_size,self.image_crop_size,1)).repeat_interleave(diff,dim=0)), dim=0)
                elif diff<0:
                    assert False
                else:
                    pass
            
            return video_feats, clean_feats, noisy_feats, clean_log_spec, \
                    noisy_log_spec, noisy_phase, length, (self.root,)+mix_name 


    def get_spectrogram(self, path):
        wav, sr = torchaudio.load(path)
        length = wav.shape[-1]
        x_stft = torch.stft(
            wav, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            center=True, 
            normalized=False, 
            onesided=True,
            pad_mode='reflect',
            return_complex=False,
            window=torch.hamming_window(self.win_length))
    #     print(x_stft.shape)
        feature = torch.norm(x_stft,dim=-1,p=2)+self.epsilon
        phase  = x_stft/feature.unsqueeze(-1)

        # feature = feature.log()
        feature = torch.log1p(feature)

        return feature[0], phase[0], length


    def __getitem__(self, index):
        video_feats, clean_feats, noisy_feats, clean_log_spec, noisy_log_spec, \
            noisy_phase, length, mix_name = self.load_feature(self.names[index])
        # clean_feats, noisy_feats, video_feats = torch.from_numpy(clean_feats.astype(np.float32)), \
        #     torch.from_numpy(noisy_feats.astype(np.float32)), torch.from_numpy(video_feats.astype(np.float32))
        if self.normalize:
            with torch.no_grad():
                clean_feats = F.layer_norm(clean_feats, clean_feats.shape[1:])
                noisy_feats = F.layer_norm(noisy_feats, noisy_feats.shape[1:])

        return {"video_feats": video_feats, "noisy_feats": noisy_feats, 
                "clean_log_spec": clean_log_spec, "noisy_log_spec": noisy_log_spec, 
                "noisy_phase": noisy_phase, "length": length, "filename": mix_name}

    def __len__(self):
        return len(self.names)

    def collater(self, samples):
        frames_video_feats = [s['video_feats'].size(0) for s in samples]
        frames_noisy_feats = [s['noisy_feats'].size(0) for s in samples]
        
        frames_clean_spec = [s['clean_log_spec'].size(0) for s in samples]
        frames_noisy_spec = [s['noisy_log_spec'].size(0) for s in samples]

        if not self.no_ssl:
            assert frames_video_feats == frames_noisy_feats
        else:
            assert frames_video_feats == frames_noisy_spec
        assert frames_clean_spec == frames_noisy_spec

        batch_video_feats = torch.zeros(len(samples), max(frames_video_feats), 
            self.image_crop_size, self.image_crop_size, 1, dtype=torch.float32)
        batch_noisy_feats = torch.zeros(len(samples), max(frames_video_feats), 
            self.F, dtype=torch.float32)

        batch_clean_spec = torch.zeros(len(samples), max(frames_noisy_spec), 
            self.n_fft//2 + 1, dtype=torch.float32)
        batch_noisy_spec = torch.zeros(len(samples), max(frames_noisy_spec), 
            self.n_fft//2 + 1, dtype=torch.float32)
        batch_noisy_phase = torch.zeros(len(samples), max(frames_noisy_spec), 
            self.n_fft//2 + 1, 2, dtype=torch.float32)
        length_lst = []
        fn_lst = []

        for i in range(len(samples)):
            batch_video_feats[i,:frames_video_feats[i]] = samples[i]['video_feats']
            batch_noisy_feats[i,:frames_noisy_feats[i]] = samples[i]['noisy_feats']
            batch_clean_spec[i,:frames_clean_spec[i]] = samples[i]['clean_log_spec']
            batch_noisy_spec[i,:frames_noisy_spec[i]] = samples[i]['noisy_log_spec']
            batch_noisy_phase[i,:frames_noisy_spec[i]] = samples[i]['noisy_phase']
            length_lst.append(samples[i]['length'])
            fn_lst.append(samples[i]['filename'])

        return batch_video_feats.permute((0, 4, 1, 2, 3)).contiguous(), batch_noisy_feats.transpose(1, 2), \
                batch_clean_spec, batch_noisy_spec, batch_noisy_phase, length_lst, fn_lst


if __name__=='__main__':

    dataset = myAVHubertDataset("../data/val.lst", no_ssl=True)
    sample = dataset[0]

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4,
        sampler=None, num_workers=0, shuffle=False,
        pin_memory=True, collate_fn=dataset.collater)

    for data in data_loader:
        video_feats, noisy_feats, clean_spec, noisy_spec, noisy_phase, length_lst, fn_lst = data
        print(f"video_feats shape: {video_feats.shape}")
        print(f"noisy_feats shape: {noisy_feats.shape}")
        print(f"clean_spec shape: {clean_spec.shape}")
        print(f"noisy_spec shape: {noisy_spec.shape}")
        print(f"noisy_phase shape: {noisy_phase.shape}")
        print(f"length list : {length_lst}")
        print(f"filename list : {fn_lst}")

        # import torchvision.models as models
        # import torch.nn as nn
        # model = models.resnet18(pretrained=True)
        # feature_extractor = nn.Sequential(*list(model.children())[:-1])
        # tmp = video_feats.permute(0,2,1,3,4).view(-1,1,88,88).repeat_interleave(3,dim=1)
        # tmp_out = feature_extractor(tmp).view(4,147,-1)



        sys.exit(0)
