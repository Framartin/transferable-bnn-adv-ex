#!/bin/bash -l

echo "*** Train 1 cSGLD for sample / cycle ***"

# launch with:
# bash train_sse_mcmc_rq_nb_samples2.sh >>log/cifar10/train_sse_mcmc_rq_nb_samples2.log 2>&1

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

DATASET="CIFAR10"
ARCH="PreResNet110"
ITER=1
BASE_DIR="../models"
DATA_PATH="../data"

# original config:
#  INJECT_NOISE_OR_NOT="--inject_noise"
#  MAX_LR=0.5
#  CYCLE_EPOCHS=50
#  CYCLE_SAVES=3
#  CYCLES=34
#  NOISE_EPOCHS=3

for NB_SAMPLES in {1..9} ; do
  # skip training with 3 samples / cycles (already done)
  if [ $NB_SAMPLES -eq 3 ]; then
      continue
  fi

  INJECT_NOISE_OR_NOT="--inject_noise"
  MAX_LR=0.5
  CYCLE_EPOCHS=$(( 47 + $NB_SAMPLES ))
  CYCLES=5
  WD=3e-4

  python sse_mcmc_train.py $INJECT_NOISE_OR_NOT \
      --dir="${BASE_DIR}/${DATASET}/${ARCH}/RQ/cSGLD_cycles${CYCLES}_savespercycle${NB_SAMPLES}" \
      --model="$ARCH" --dataset="$DATASET" --noise_epochs="$NB_SAMPLES" --data_path="$DATA_PATH" \
      --alpha=1 --cycles=$CYCLES --iter="$ITER" \
      --cycle_epochs=$CYCLE_EPOCHS --cycle_saves="$NB_SAMPLES" --max_lr=$MAX_LR --wd=$WD \
      --device_id 0 --transform="NoNormalization"

done
