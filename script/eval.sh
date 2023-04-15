#================================Single Task================================#

python idea.py --data "TAOP" --mode eval --resume archive/checkpoints/TAOP/model_best.pth --num_workers 4

python idea.py --data "APTOS" --mode eval --resume archive/checkpoints/APTOS/model_best.pth --num_workers 4

python idea.py --data "DDR" --mode eval --resume archive/checkpoints/DDR/model_best.pth --num_workers 4
 
python idea.py --data "AMD" --mode eval --resume archive/checkpoints/AMD/model_best.pth --num_workers 4

python idea.py --data "LAG" --mode eval --resume archive/checkpoints/LAG/model_best.pth --num_workers 4

python idea.py --data "PALM" --mode eval --resume archive/checkpoints/PALM/model_best.pth --num_workers 4

python idea.py --data "REFUGE" --mode eval --resume archive/checkpoints/REFUGE/model_best.pth --num_workers 4

python idea.py --data "ODIR-5K" --mode eval --resume archive/checkpoints/ODIR-5K/model_best.pth --num_workers 4

python idea.py --data "RFMiD" --mode eval --resume archive/checkpoints/RFMiD/model_best.pth --num_workers 4

python idea.py --data "DR+" --mode eval --resume archive/checkpoints/DR+/model_best.pth --num_workers 4


python idea.py --data "IDRiD" --mode eval --resume archive/checkpoints/IDRiD/model_best.pth --num_workers 4

#================================Multi Task================================#

CUDA_VISIBLE_DEVICES=1 python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+"  --mode eval --method HPS --multi_task --num_workers 4 --resume archive/checkpoints/HPS/model_best.pth

CUDA_VISIBLE_DEVICES=1 python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+"  --mode eval --method MTAN --multi_task --num_workers 4 --resume archive/checkpoints/MTAN/model_best.pth