#!/bin/bash -l
#should take XX days
# launch with:
# bash scripts/imagenet/train_dnn_densenet.sh >>log/imagenet/run_train_dnn_densenet.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=1


DATAPATH="../data/ILSVRC2012"
ARCH="densenet121"
LR=0.1
EPOCHS=130
BATCH_SIZE=64
WORKERS=10
PRINT_FREQ=400
DIR="models/ImageNet/${ARCH}/single_model"


date

python -u train_dnn_imagenet.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --epochs $EPOCHS --gpu $CUDA_VISIBLE_DEVICES

# no fixed seed to speed up

date