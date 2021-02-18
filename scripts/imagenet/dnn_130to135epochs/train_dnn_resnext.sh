#!/bin/bash -l
# launch with:
# bash scripts/imagenet/dnn_130to135epochs/train_dnn_resnext.sh >>log/imagenet/dnn_130to135epochs/run_train_dnn_resnext.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=1


DATAPATH="../data/ILSVRC2012"
ARCH="resnext50_32x4d"
LR=0.1
EPOCHS=135
BATCH_SIZE=64
WORKERS=10
PRINT_FREQ=400
DIR="models/ImageNet/${ARCH}/single_model"


date

python -u train_dnn_imagenet.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --epochs $EPOCHS --gpu $CUDA_VISIBLE_DEVICES \
  --resume "models/ImageNet/resnext50_32x4d/single_model/checkpoints/ImageNet-SGD_resnext50_32x4d_lr0.1_ep00129.pt.tar"

# no fixed seed to speed up

date