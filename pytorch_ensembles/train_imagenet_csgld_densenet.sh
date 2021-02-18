#!/bin/bash -l
#should take 13.5 days
# launch with:
# bash train_imagenet_csgld_densenet.sh >>log/run_train_imagenet_csgld_densenet.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=1


DATAPATH="../../data/ILSVRC2012"
ARCH="densenet121"
LR=0.1
CYCLES=3
SAMPLES_PER_CYCLE=3
BATCH_SIZE=64
WORKERS=10
PRINT_FREQ=400
DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}"


date

# densenet121 should take 2h24 / epoch on 1 V100

python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --gpu $CUDA_VISIBLE_DEVICES

# no fixed seed to speed up

date