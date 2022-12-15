#CUDA_VISIBLE_DEVICES=1 python idea.py --mode eval --data ODIR-5K --resume archive/checkpoints/ODIR-5K/baseline.pth
python idea.py --mode eval --data APTOS --resume archive/checkpoints/APTOS/model_best.pth
#Micro F1 Score: 0.7811447811447811, Macro F1 Score: 0.7736465847333169
#python idea.py --mode eval --data RFMiD --resume archive/checkpoints/RFMiD/baseline.pth
#Micro F1 Score: 0.7579737335834896, Macro F1 Score: 0.22601375793198472, Samples F1 Score: 0.5819977678571429
#python idea.py --mode eval --data Kaggle --resume archive/checkpoints/Kaggle/model_best.pth
# Micro F1 Score: 0.7091295116772824, Macro F1 Score: 0.6009858310022451, Samples F1 Score: 0.6907166907166907

# CUDA_VISIBLE_DEVICES=1 python idea.py --mode eval --multi_task --batch_size 64 --data "ODIR-5K, TAOP, RFMiD, KaggleDR+, Kaggle, APTOS" --resume archive/checkpoints/Hard_Params/model_best.pth
