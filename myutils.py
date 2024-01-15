from collections import defaultdict, deque, OrderedDict
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os

import numpy as np
import math

def save_path_formatter(args):

    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = [data_folder_name]

    key_map = OrderedDict()
    # key_map['epochs'] = 'ep'
    key_map['optimizer']=''
    key_map['lr']=''
    key_map['loss']=''
    # key_map['batch_size']='bs'
    key_map['seed']='seed'
    key_map['ckpt_path'] = ''
    key_map['freeze_extractor']='freeze_extractor'
    key_map['no_finetune']='no_finetune'
    key_map['from_scratch']='from_scratch'
    key_map['no_ssl']='no_ssl'
    key_map['no_video']='no_video'
    key_map['mapping']='mapping'
    key_map['crossfeat']='crossfeat'
    

    for key, key2 in key_map.items():
        value = args_dict[key]
        if key == 'ckpt_path':
            value = os.path.basename(value).split(".")[0]
        if key2 != '':
            folder_string.append('{}.{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M")
    return os.path.join('logs_lowresource_mandarin',save_path,timestamp).replace("\\","/")    



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)




from pesq import pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources

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
    s_snr  = si_snr(enhanced,clean)
    s_sdr  = bss_eval_sources(clean,enhanced,False)[0][0]
    if math.isnan(s_pesq):
        s_pesq=0
    if math.isnan(s_stoi):
        s_stoi=0
    return round(s_pesq,5), round(s_stoi,5), round(s_snr,5), round(s_sdr,5)



def si_snr(x, s, remove_dc=True):

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


n_fft   = 512
win_length = 400
hop_length = 160
epsilon  = 1e-6

def feature_to_wav(x, length):

  fea, phase = x
  device = fea.device
  fea = torch.expm1(fea)
  fea = phase*(fea-epsilon).unsqueeze(-1)
    
  wav = torch.istft(
      fea, 
      n_fft=n_fft, 
      hop_length=hop_length, 
      win_length=win_length, 
      center=True, 
      normalized=False, 
#       onesided=True, 
      window=torch.hamming_window(win_length).to(device),
      return_complex=False,
      length=length
    )
  return wav
