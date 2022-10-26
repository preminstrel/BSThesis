CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data ODIR-5K --batch_size 160 --save_freq 50 --valid_freq 1 --use_wandb --project ODIR-5K > ODIR-5K.log
# nohup python idea.py --data TAOP --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project TAOP > TAOP.log &
# python idea.py --data ODIR-5K --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project ODIR-5K
nohup python idea.py --data RFMiD --batch_size 160 --save_freq 50 --valid_freq 1 --use_wandb --project RFMiD > RFMiD.log