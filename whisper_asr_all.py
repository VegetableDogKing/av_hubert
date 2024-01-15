# -*- coding: Big5 -*-
import os
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm

#Install whisper using: pip install -U openai-whisper
#https://github.com/openai/whisper

def get_filepaths(directory, format='.wav'):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith(format):
                        file_paths.append(filepath)  
      return file_paths 

input_dir = '/home/richardleelai/jchou/project/avhubert_avse/code/enhan_result/TMHINT,AdamW,0.001,l1,seed.999,freeze_extractor.False,no_finetune.False,from_scratch.False,no_ssl.True,no_video.False,mapping.False/12-22-11.19/SP05/audio/noisy_enh/test/babble_2db/'
file_list = get_filepaths(input_dir, format='.wav') #loop all the .wav file in dir

df =  pd.DataFrame(columns=['wavname','transcript'])
model = whisper.load_model("base")


for path in tqdm(file_list):

      filename = os.path.basename(path)
      result = model.transcribe(path)
      for seg in result['segments']:
            transcript = seg['text']
            start = seg['start']
            end = seg['end']
            results = pd.DataFrame([{'wavname':filename,'start':start,'end':end,'transcript':transcript}])
            df = pd.concat([df, results])

outputname = input_dir.split(os.sep)[-1]
df.to_csv(outputname+'_whisper.csv', sep=',', index=False)




