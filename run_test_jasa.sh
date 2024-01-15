

## AVSE_SSL (checkpoint:lrs3+voxceleb2)
python test.py \
--resume logs/TMHINT,AdamW,0.0001,l1,seed.999,base_vox_iter5,freeze_extractor.True,no_finetune.False,from_scratch.False,no_ssl.False,no_video.False,mapping.False,crossfeat.True/02-09-15.12/bestmodel.pth \
--crossfeat \
--test-list test3.lst \

## AVSE_SSL (checkpoint:lrs3+voxceleb2+noise)
python test.py \
--resume logs/TMHINT,AdamW,0.0001,l1,seed.999,base_vox_noise_iter5,freeze_extractor.True,no_finetune.False,from_scratch.False,no_ssl.False,no_video.False,mapping.False,crossfeat.True/02-09-15.24/bestmodel.pth \
--crossfeat \
--test-list test3.lst\



##AVSE_SSL
#python test.py \
#--resume logs/TMHINT,AdamW,0.0001,l1,seed.999,freeze_extractor.False,no_finetune.False,from_scratch.False,no_ssl.False,mapping.False/06-05-15.08/bestmodel.pth \
#--test-list test3.lst \


## AVSE
#python test.py \
#--resume logs/TMHINT,AdamW,0.001,l1,seed.999,freeze_extractor.False,no_finetune.False,from_scratch.False,no_ssl.True,no_video.False,mapping.False/12-22-11.19/bestmodel.pth \
#--test-list test3.lst \
#--no-ssl \

## AOSE
#python test.py \
#--resume logs/TMHINT,AdamW,0.0001,l1,seed.999,freeze_extractor.False,no_finetune.False,from_scratch.False,no_ssl.True,no_video.True,mapping.False/11-12-10.03/bestmodel.pth  \
#--test-list test3.lst \
#--no-video \
#--no-ssl \

