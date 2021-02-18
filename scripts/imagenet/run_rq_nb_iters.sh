#!/bin/bash -l
# bash scripts/imagenet/run_rq_nb_iters.sh >>log/imagenet/run_rq_nb_iters.log 2>&1

echo "tranferability wrt number of iterations"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

PATH_CSV="X_adv/ImageNet/RQ/results_nb_iters.csv"
DATAPATH="../data/ILSVRC2012"
TARGET="ImageNet/pretrained/resnet50"

ATTACK="python -u attack_csgld_pgd_torch.py"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
# step=1/4 eps
#ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.75"
#ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.00392"

NB_EXAMPLES=2500
ARGS_COMMON="--n-examples $NB_EXAMPLES --csv-export ${PATH_CSV} --export-target-per-iter --model-target-path $TARGET --data-path $DATAPATH --seed 42 --no-save"

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0

# cSGLD
PATH_BNN="./models/ImageNet/resnet50/cSGLD_cycles5_samples3_bs256/"
echo "-- I-F(S)GM - 1K examples, 200 iterations, no ensemble (L2 and Linf norms), no random init --"
$ATTACK $PATH_BNN $ARGS_COMMON --n-iter 200 --shuffle --batch-size 128 $ARGS_L2
print_time
$ATTACK $PATH_BNN $ARGS_COMMON --n-iter 200 --shuffle --batch-size 128 $ARGS_Linf
print_time

echo "-- MI-F(S)GM - 1K examples, 200 iterations, no ensemble (L2 and Linf norms), no random init --"
MOMENTUM=0.9
$ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 200 --shuffle --batch-size 128 $ARGS_L2
print_time
$ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 200 --shuffle --batch-size 128 $ARGS_Linf
print_time

echo "-- PGD light - **1** random init, 200 iterations, no ensemble (L2 and Linf norms) --"
$ATTACK $PATH_BNN $ARGS_COMMON --n-random-init 1 --n-iter 200 --shuffle --batch-size 128 $ARGS_L2
print_time
$ATTACK $PATH_BNN $ARGS_COMMON --n-random-init 1 --n-iter 200 --shuffle --batch-size 128 $ARGS_Linf
print_time

# DNNs
for NB_DNN in 1 2 5 15 ; do
  echo "******** ENSEMBLE of $NB_DNN DNNs ********"
  PATH_DNNS="./models/ImageNet/resnet50/deepens_imagenet"
  echo "-- I-F(S)GM - 200 iterations, no ensemble (L2 and Linf norms), no random init --"
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle --batch-size 128 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle --batch-size 128 $ARGS_Linf
  print_time

  echo "-- MI-F(S)GM - 200 iterations, no ensemble (L2 and Linf norms), no random init --"
  MOMENTUM=0.9
  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle --batch-size 128 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle --batch-size 128 $ARGS_Linf
  print_time

  echo "-- PGD light - **1** random init, 200 iterations, no ensemble (L2 and Linf norms) --"
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 1 --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle --batch-size 128 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 1 --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle --batch-size 128 $ARGS_Linf
  print_time

done

