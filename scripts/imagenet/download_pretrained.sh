#!/bin/bash -l
#SBATCH --time=0-03:00:00 # 3 hours
#SBATCH -J DLmodels  # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 1              # 1 cores per task
#SBATCH -o "log/imagenet/download_pretrained_%j.log"

echo "Download ImageNet pretrained models from pytorch-ensembles"
echo "See https://github.com/bayesgroup/pytorch-ensembles"

command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate
pip install wldhx.yadisk-direct

cd models/ImageNet/resnet50 || exit

# download deep ensemble resnet50 models
curl -L $(yadisk-direct https://yadi.sk/d/rdk6ylF5mK8ptw?w=1) -o deepens_imagenet.zip
unzip deepens_imagenet.zip

# trained with https://github.com/bayesgroup/pytorch-ensembles/blob/master/train/imagenet/train_imagenet.py