# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data TAOP --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project TAOP > logs/TAOP.log &
# nohup python idea.py --data ODIR-5K --batch_size 160 --save_freq 50 --lr 3e-4 --valid_freq 1 --project ODIR-5K --use_wandb > logs/ODIR-5K.log
# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data RFMiD --batch_size 160 --save_freq 50 --valid_freq 1 --use_wandb --project RFMiD --lr 5e-4 > logs/RFMiD.log

# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, KaggleDR+" --resume archive/checkpoints/Hard_Params/model_best.pth --project Hard-Params --multi_task --valid_freq 1 --epochs 400 --use_wandb --num_workers 8 > archive/logs/Hard_Params.log

nohup python idea.py --data ODIR-5K --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project ODIR-5K > archive/logs/ODIR-5K/balanced.log &

# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data Kaggle --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project Kaggle --lr 3e-4 > archive/logs/Kaggle/baseline_balanced.log

#python idea.py --data KaggleDR+ --batch_size 128 --resume archive/checkpoints/KaggleDR+/baseline.pth --save_freq 50 --valid_freq 1 --project KaggleDR+ --lr 1e-4

python idea.py --data ODIR-5K --batch_size 128 --save_freq 50 --valid_freq 1 --project ODIR-5K