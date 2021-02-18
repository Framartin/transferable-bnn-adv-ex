#!/bin/bash -l
# launch with:
# bash scripts/imagenet/dnn_130to135epochs/train_dnn_resnet50.sh >>log/imagenet/dnn_130to135epochs/run_train_dnn_resnet50.log 2>&1

echo
echo "resume resnet50 from a random model in deepens models"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=1


DATAPATH="../data/ILSVRC2012"
ARCH="resnet50"
LR=0.1
EPOCHS=135
BATCH_SIZE=256
WORKERS=10
PRINT_FREQ=400
DIR="models/ImageNet/${ARCH}/single_model"


date

python -u train_dnn_imagenet.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --epochs $EPOCHS \
  --resume "models/ImageNet/resnet50/deepens_imagenet/ImageNet-ResNet50-052e7f78e4db--1564492444-1.pth.tar"

# no fixed seed to speed up

date