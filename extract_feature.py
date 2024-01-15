import cv2
import tempfile
import torch
import os, sys
sys.path.append('./av_hubert/avhubert')
import utils as avhubert_utils
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from IPython.display import HTML
from myhubert_dataset import myAVHubertDataset
import pdb

def extract_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  transform = avhubert_utils.Compose([
      avhubert_utils.Normalize(0.0, 255.0),
      avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
      avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
  frames = avhubert_utils.load_video(video_path)
  print(f"Load video {video_path}: shape {frames.shape}")
  frames = transform(frames)
  print(f"Center crop video to: {frames.shape}")
  frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
  print(f"Frame shape: {frames.shape}")
  model = models[0]
  if hasattr(models[0], 'decoder'):
    print(f"Checkpoint: fine-tuned")
    model = models[0].encoder.w2v_model
  else:
    print(f"Checkpoint: pre-trained w/o fine-tuning")
  model.cuda()
  model.eval()
  with torch.no_grad():
    # Specify output_layer if you want to extract feature of an intermediate layer
    feature, _ = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
    feature = feature.squeeze(dim=0)
  print(f"Video feature shape: {feature.shape}")
  return feature

def extract_audio_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  dataset = myAVHubertDataset("../data/train.lst")
  frames, noisy_audio = dataset[0]['video_feats'], dataset[0]['noisy_feats']
  frames = frames.unsqueeze(dim=0).permute((0, 4, 1, 2, 3)).contiguous().cuda()
  noisy_audio = noisy_audio.unsqueeze(dim=0).transpose(1, 2).cuda()

  model = models[0]
  if hasattr(models[0], 'decoder'):
    print(f"Checkpoint: fine-tuned")
    model = models[0].encoder.w2v_model
  else:
    print(f"Checkpoint: pre-trained w/o fine-tuning")
  model.cuda()
  model.eval()
  with torch.no_grad():
    # Specify output_layer if you want to extract feature of an intermediate layer
    feature, _ = model.extract_finetune(source={'video': frames, 'audio': noisy_audio}, padding_mask=None, output_layer=None)
    feature = feature.squeeze(dim=0)
  print(f"Audio-Video feature shape: {feature.shape}")
  return feature

mouth_roi_path = "/home/jchou/project/avhubert_avse/corpus/mandarin/SP01/video/mouth/SP01_001.mp4"
ckpt_path = "/home/jchou/project/avhubert_avse/code/checkpoint/base_lrs3_iter5.pt" ## Pretrained Models
# ckpt_path = "/home/jchou/project/avhubert_avse/code/checkpoint/base_vox_433h.pt" ## Finetuned Models for Visual Speech Recognition
# ckpt_path = "/home/jchou/project/avhubert_avse/code/checkpoint/base_noise_pt_noise_ft_433h.pt" ## Finetuned Models for Audio-Visual Speech Recognition 
user_dir = "/home/jchou/project/avhubert_avse/code/av_hubert/avhubert"
# feature = extract_visual_feature(mouth_roi_path, ckpt_path, user_dir)
feature = extract_audio_visual_feature(mouth_roi_path, ckpt_path, user_dir)
