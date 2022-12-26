#============================================Single Task============================================#

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP" --use_wandb --project TAOP > archive/logs/TAOP/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "APTOS" --use_wandb --project APTOS > archive/logs/APTOS/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "DDR" --use_wandb --project DDR > archive/logs/DDR/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "AMD" --use_wandb --project AMD > archive/logs/AMD/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "LAG" --use_wandb --project LAG > archive/logs/LAG/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "PALM" --use_wandb --project PALM > archive/logs/PALM/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "REFUGE" --use_wandb --project REFUGE > archive/logs/REFUGE/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K" --use_wandb --project ODIR-5K > archive/logs/ODIR-5K/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "RFMiD" --use_wandb --project RFMiD > archive/logs/RFMiD/baseline.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "DR+" --use_wandb --project DR+ > archive/logs/DR+/baseline.log &

#============================================Multi Task============================================#
nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project HPS --method HPS --multi_task --epochs 1000 --num_workers 8 --use_wandb > archive/logs/HPS/baseline.log &

nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project MTAN --method MTAN --multi_task --epochs 1000 --num_workers 8 --use_wandb > archive/logs/MTAN/baseline.log &



#============================================Misc============================================#

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MMoE --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method MMoE --use_wandb --batch_size 12 --batches 2000 > archive/logs/MMoE/unified.log & 

nohup python idea.py --data "TAOP" --project MMoE --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method MMoE --use_wandb --batch_size 10 --batches 2000 > archive/logs/MMoE/unified.log & 

nohup python idea.py --data "TAOP, APTOS, DDR, PALM, LAG, AMD, REFUGE, ODIR-5K, RFMiD, DR" --project HPS --method HPS --multi_task --valid_freq 1 --epochs 400 --use_wandb --preflight > archive/logs/HPS/baseline.log

python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project CGC --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method CGC --batch_size 4 --batches 2000

python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MTAN --multi_task --valid_freq 1 --lr 1e-6 --epochs 400 --method MTAN --batch_size 4 --batches 2000

nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project MTAN --multi_task --valid_freq 1 --lr 1e-4 --epochs 400 --method MTAN --batch_size 128 --batches 200 --use_wandb --num_workers 8 > archive/logs/MTAN/unified.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K, TAOP, RFMiD, APTOS, Kaggle, DR+" --project CGC --multi_task --valid_freq 1 --lr 1e-6 --epochs 4000 --use_wandb --method CGC --batch_size 16 --batches 1600 --resume archive/checkpoints/CGC/epoch_400.pth > archive/logs/CGC/unified.log &

nohup python idea.py --data "TAOP, APTOS, DDR, PALM, LAG, AMD, REFUGE, ODIR-5K, RFMiD, DR+" --project MMoE --multi_task --valid_freq 1 --lr 1e-6 --epochs 2000 --use_wandb --method MMoE --batch_size 12 --batches 4000 > archive/logs/MMoE/unified.log &