#============================================Single Task============================================#

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP" --use_wandb --project TAOP > archive/logs/TAOP/preprocess.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "APTOS" --use_wandb --project APTOS > archive/logs/APTOS/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "APTOS" --use_wandb --preprocessed --project APTOS > archive/logs/APTOS/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "DDR" --use_wandb --project DDR > archive/logs/DDR/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "DDR" --use_wandb --preprocessed --project DDR > archive/logs/DDR/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "AMD" --use_wandb --project AMD > archive/logs/AMD/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "AMD" --use_wandb --preprocessed --project AMD > archive/logs/AMD/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "LAG" --use_wandb --project LAG > archive/logs/LAG/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "LAG" --use_wandb --preprocessed --project LAG > archive/logs/LAG/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "PALM" --use_wandb --project PALM > archive/logs/PALM/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "PALM" --use_wandb --preprocessed --project PALM > archive/logs/PALM/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "REFUGE" --use_wandb --project REFUGE > archive/logs/REFUGE/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "REFUGE" --use_wandb --preprocessed --project REFUGE > archive/logs/REFUGE/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K" --use_wandb --project ODIR-5K > archive/logs/ODIR-5K/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "ODIR-5K" --use_wandb --preprocessed --project ODIR-5K > archive/logs/ODIR-5K/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "RFMiD" --use_wandb --project RFMiD > archive/logs/RFMiD/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "RFMiD" --use_wandb --preprocessed --project RFMiD > archive/logs/RFMiD/preprocessed.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "DR+" --use_wandb --project DR+ > archive/logs/DR+/baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "DR+" --use_wandb --preprocessed --project DR+ > archive/logs/DR+/preprocessed.log &

#============================================Multi Task============================================#

# HPS Model 23.6822M (23.6822M trainable)
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project HPS --method HPS --multi_task --epochs 1000 --num_workers 4 --use_wandb > archive/logs/HPS/pretrained.log &

# MTAN Model 73.3729M (73.3729M trainable)
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project MTAN --method MTAN --multi_task --epochs 1000 --num_workers 2 --use_wandb > archive/logs/MTAN/baseline.log &

# MMoE Model 75.2141M (75.2141M trainable)
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project MMoE --method MMoE --multi_task --epochs 1000 --num_workers 16 --batch_size 12 --batches 300 --accumulate --use_wandb > archive/logs/MMoE/baseline_accumulate_random.log &

# CGC Model 261.7731M (261.7731M trainable)
CUDA_VISIBLE_DEVICES=1 python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project CGC --method CGC --multi_task --epochs 1000 --num_workers 8 --batch_size 13 --accumulate --use_wandb > archive/logs/CGC/baseline.log & 

# DSelectK Model 70.6983M (70.6983M trainable)
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project DSelectK --method DSelectK --multi_task --epochs 1000 --num_workers 16 --batch_size 12 --batches 300 --accumulate --use_wandb > archive/logs/DSelectK/baseline_accumulate.log &

# LTB Model 235.2551M (235.2551M trainable)
CUDA_VISIBLE_DEVICES=1 python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project LTB --method LTB --multi_task --epochs 1000 --num_workers 2 --batch_size 32 --use_wandb > archive/logs/LTB/baseline.log &

# LTB Model 235.2551M (235.2551M trainable)
CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project Adapter --method Adapter --multi_task --epochs 1000 --batches 500 --num_workers 4 --batch_size 128 --use_wandb > archive/logs/Adapter/series_adapters.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project Adapter --method Adapter --multi_task --epochs 1000 --batches 500 --num_workers 4 --batch_size 128 --use_wandb > archive/logs/Adapter/parallel_adapters.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project Adapter --method Adapter --multi_task --epochs 1000 --batches 5000 --num_workers 4 --batch_size 128 --image_size 64> archive/logs/Adapter/parallel_adapters_64x64.log &

CUDA_VISIBLE_DEVICES=1 nohup python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project Adapter --method Adapter --multi_task --epochs 1000 --batches 300 --num_workers 4 --batch_size 100 --use_wandb > archive/logs/Adapter/parallel_adapters_new.log &

#============================================Pretrained ResNet50============================================#

# MoCo v2
CUDA_VISIBLE_DEVICES=1 nohup python moco_train.py --mlp --moco-t 0.2 --aug-plus --cos > archive/logs/moco/moco_v2.log