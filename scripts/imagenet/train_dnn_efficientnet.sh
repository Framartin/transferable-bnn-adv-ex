#!/bin/bash -l
# launch with:
# bash scripts/imagenet/train_dnn_efficientnet.sh >>log/imagenet/run_train_dnn_efficientnet.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=2


DATAPATH="../data/ILSVRC2012"
ARCH="efficientnet_b0"
LR=0.1
EPOCHS=130
BATCH_SIZE=256
WORKERS=10
PRINT_FREQ=400
DIR="models/ImageNet/${ARCH}/single_model"


date

# mnasnet should take 1h / epoch on 1 V100

python -u train_dnn_imagenet.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --epochs $EPOCHS --gpu $CUDA_VISIBLE_DEVICES

# no fixed seed to speed up

date