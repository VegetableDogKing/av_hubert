### AVSE (w/o SSL) 
python main.py \
--seed 999 \
--start-epoch 0 \
--epochs 50 \
--lr 0.0001 \
--loss l1 \
--optimizer AdamW \
--no-ssl \
--train-list train_60k.lst \
--val-list val_6000.lst \



