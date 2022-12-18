# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data TAOP --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project TAOP > logs/TAOP.log &
# nohup python idea.py --data ODIR-5K --batch_size 160 --save_freq 50 --lr 3e-4 --valid_freq 1 --project ODIR-5K --use_wandb > logs/ODIR-5K.log
# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data RFMiD --batch_size 160 --save_freq 50 --valid_freq 1 --use_wandb --project RFMiD --lr 5e-4 > logs/RFMiD.log

# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --balanced_sampling --resume archive/checkpoints/Hard_Params/model_best.pth --project Hard-Params --multi_task --valid_freq 1 --epochs 400 --use_wandb --num_workers 8 > archive/logs/Hard_Params.log

# nohup python idea.py --data DR+ --batch_size 128 --save_freq 500 --valid_freq 1 --use_wandb --project KaggleDR+ > archive/logs/DR+/balanced.log &

# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data Kaggle --batch_size 128 --save_freq 50 --valid_freq 1 --use_wandb --project Kaggle --lr 3e-4 > archive/logs/Kaggle/baseline_balanced.log

#CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data DR+ --batch_size 128 --save_freq 500 --valid_freq 1 --project DR+ --lr 3e-4 --use_wandb --num_workers 8 > archive/logs/DR+/baseline.log &

# python idea.py --data ODIR-5K --batch_size 128 --save_freq 50 --valid_freq 1 --project ODIR-5K

# CUDA_VISIBLE_DEVICES=1 python idea.py --data "ODIR-5K, RFMiD, DR+" --balanced_sampling  --project Hard-Params --multi_task --valid_freq 1 --epochs 400 --num_workers 8

# CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MMoE --multi_task --valid_freq 1 --epochs 400 --num_workers 8 --method MMoE --use_wandb > archive/logs/MMoE/unified.log &

#=============================MoCo======================================#
python main.py 

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MMoE --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method MMoE --use_wandb --batch_size 12 --batches 2000 > archive/logs/MMoE/unified.log & 

nohup python idea.py --data "TAOP" --project MMoE --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method MMoE --use_wandb --batch_size 10 --batches 2000 > archive/logs/MMoE/unified.log & 

nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project Hard-Params --balanced_sampling --method HPS --multi_task --valid_freq 1 --epochs 400 --use_wandb --num_workers 8 > archive/logs/Hard_Params/balanced.log

python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project CGC --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method CGC --batch_size 4 --batches 2000

nohup python idea.py --data TAOP --batch_size 128 --save_freq 500 --valid_freq 1 --multi_gpus --use_wandb --project TAOP > logs/TAOP/baseline.log &

python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MTAN --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method MTAN --batch_size 4 --batches 2000

nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MTAN --multi_task --valid_freq 1 --lr 1e-4 --epochs 400 --method MTAN --batch_size 128 --batches 200 --use_wandb --num_workers 8 > archive/logs/MTAN/unified.log &
