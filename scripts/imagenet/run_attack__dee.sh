#!/bin/bash -l
# bash scripts/imagenet/run_attack__dee.sh >>log/imagenet/run_attack__dee.log 2>&1

echo "\n RQ: nb of DNNs to have same transferability than cSGLD \n"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

PATH_CSV="X_adv/ImageNet/resnet50/results_dee.csv"
DATAPATH="../data/ILSVRC2012"
TARGET="ImageNet/pretrained/resnet50"

ATTACK="python -u attack_csgld_pgd_torch.py"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"

NB_EXAMPLES=5000

ARGS_COMMON="--n-examples $NB_EXAMPLES --shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --data-path $DATAPATH --seed 42 --no-save"


print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0

# DNNs
for NB_DNN in {1..15} ; do
  echo "******** Ensemble of $NB_DNN DNNs ********"
  PATH_DNNS="./models/ImageNet/resnet50/deepens_imagenet --n-models-cycle 1 --limit-n-cycles $NB_DNN"

  echo "-- I-F(S)GM light - 1K examples, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-iter 50 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-iter 50 $ARGS_Linf
  print_time

  echo "-- MI-F(S)GM light - momentum 0.9, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
  MOMENTUM=0.9
  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_Linf
  print_time

  ARGS_FULL_ITER="--n-ensemble $NB_DNN --batch-size 16"

  ARGS_L2_FSGM="--norm 2 --max-norm 3 --norm-step 3"
  ARGS_Linf_FSGM="--norm inf --max-norm 0.01568 --norm-step 0.01568"
  echo "-- F(S)GM - 1 iteration, ensemble all models, no random init --"
  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_L2_FSGM
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_Linf_FSGM
  print_time

done


# only run PGD for some values (for performance)
echo "****** PGD light - 5 random init, 50 iterations, no ensemble (L2 and Linf norms) ******"
for NB_DNN in {4..5} ; do
  echo "-- Ensemble of $NB_DNN DNNs --"
  PATH_DNNS="./models/ImageNet/resnet50/deepens_imagenet --n-models-cycle 1 --limit-n-cycles $NB_DNN"

  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_Linf
  print_time

done
