#!/bin/bash -l
# launch with:
# bash scripts/cifar10/run_rq_nb_iters.sh >>log/cifar10/run_rq_nb_iters.log 2>&1

echo "tranferability wrt number of iterations"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=3

PATH_CSV="X_adv/CIFAR10/RQ/results_nb_iters.csv"
TARGET="models_target/CIFAR10/PreResNet110/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1.pth"

ATTACK="python -u attack_csgld_pgd_torch.py"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 0.5 --norm-step 0.05"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"

ARGS_COMMON="--csv-export ${PATH_CSV} --export-target-per-iter --model-target-path $TARGET --seed 42 --no-save --batch-size 256"


print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0

# cSGLD
PATH_BNN="models/CIFAR10/PreResNet110/cSGLD_cycles5_savespercycle3_it1"
echo "-- I-F(S)GM - 1K examples, 200 iterations, no ensemble (L2 and Linf norms), no random init --"
$ATTACK $PATH_BNN $ARGS_COMMON --n-iter 200 --shuffle $ARGS_L2
print_time
$ATTACK $PATH_BNN $ARGS_COMMON --n-iter 200 --shuffle $ARGS_Linf
print_time

echo "-- MI-F(S)GM light - momentum 0.9, 200 iterations, no ensemble (L2 and Linf norms), no random init --"
MOMENTUM=0.9
$ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 200 --shuffle $ARGS_L2
print_time
$ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 200 --shuffle $ARGS_Linf
print_time

echo "-- PGD light - **1** random init, 200 iterations, no ensemble (L2 and Linf norms) --"
$ATTACK $PATH_BNN $ARGS_COMMON --n-random-init 1 --n-iter 200 --shuffle $ARGS_L2
print_time
$ATTACK $PATH_BNN $ARGS_COMMON --n-random-init 1 --n-iter 200 --shuffle $ARGS_Linf
print_time


# DNNs
for NB_DNN in 1 2 5 15 ; do
  echo "******** ENSEMBLE of $NB_DNN DNNs ********"
  PATH_DNNS="models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed50"
  echo "-- I-F(S)GM - 1K examples, 200 iterations, no ensemble (L2 and Linf norms), no random init --"
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle $ARGS_Linf
  print_time

  echo "-- MI-F(S)GM - 1K examples, 200 iterations, no ensemble (L2 and Linf norms), no random init --"
  MOMENTUM=0.9
  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle $ARGS_Linf
  print_time

  echo "-- PGD light - **1** random init, 200 iterations, no ensemble (L2 and Linf norms) --"
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 1 --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 1 --n-models-cycle 1 --limit-n-cycles $NB_DNN --n-iter 200 --shuffle $ARGS_Linf
  print_time

done
