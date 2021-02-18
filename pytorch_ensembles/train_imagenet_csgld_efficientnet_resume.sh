#!/bin/bash -l
#SBATCH --time=6-12:00:00 # approx 110 hours / cycle
#SBATCH --partition=gpu   # Use the batch partition reserved for passive jobs
#SBATCH --qos=long
#SBATCH -J cSGLDeffnetb0  # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 8              # 8 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_imagenet_csgld_efficientnet_resume_%j.log"
#SBATCH --mail-type=begin,end,fail

echo
echo "single GPU training"
echo

command -v module >/dev/null 2>&1 && module load lang/Python system/CUDA
source ../venv/bin/activate
set -x

DATAPATH="/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/"
ARCH="efficientnet_b0"
LR=0.1
CYCLES=3
SAMPLES_PER_CYCLE=3
BATCH_SIZE=64
WORKERS=8
PRINT_FREQ=400
DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}"

date

# efficientnet B0 should take 1h16 / epoch on 1 V100
python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --resume "${DIR}/ImageNet-cSGLD_efficientnet_b0_00044.pt.tar"

# no fixed seed to speed up

date