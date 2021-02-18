#!/bin/bash -l
# bash scripts/cifar10/run_attack_same_arch.sh >>log/cifar10/run_attack_same_arch.log 2>&1

echo "\n CIFAR10 - Tranferability of different attacks to the same arch \n"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0

PATH_CSV="X_adv/CIFAR10/PreResNet110/results_same_arch.csv"
TARGET="models_target/CIFAR10/PreResNet110/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1.pth"

ATTACK="python -u attack_csgld_pgd_torch.py"

# step=1/10 eps
ARGS_L2="--norm 2 --max-norm 0.5 --norm-step 0.05"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"

ARGS_COMMON="--shuffle --csv-export ${PATH_CSV} --model-target-path $TARGET --seed 42 --no-save"

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0

# cSGLD

echo "******** 15 cSGLD samples ********"
PATH_BNN="models/CIFAR10/PreResNet110/cSGLD_cycles5_savespercycle3_it1"

#echo "-- I-F(S)GM light - 50 iterations, no ensemble (L2 and Linf norms), no random init --"
#$ATTACK $PATH_BNN $ARGS_COMMON --n-iter 50 $ARGS_L2
#print_time
#$ATTACK $PATH_BNN $ARGS_COMMON --n-iter 50 $ARGS_Linf
#print_time
#
#echo "-- MI-F(S)GM light - momentum 0.9, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
#MOMENTUM=0.9
#$ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_L2
#print_time
#$ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_Linf
#print_time
#
#echo "-- PGD light - 5 random init, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
#$ATTACK $PATH_BNN $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_L2
#print_time
#$ATTACK $PATH_BNN $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_Linf
#print_time

ARGS_FULL_ITER="--n-ensemble 15 --batch-size 100"

#echo "-- F(S)GM - 1 iteration, ensemble all models, no random init --"
#ARGS_L2_FSGM="--norm 2 --max-norm 0.5 --norm-step 0.5"
#ARGS_Linf_FSGM="--norm inf --max-norm 0.01568 --norm-step 0.01568"
#$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_L2_FSGM
#print_time
#$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_Linf_FSGM
#print_time
#
#echo "-- I-F(S)GM full - 50 iterations, ensemble all models, no random init --"
#$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --n-iter 50 $ARGS_L2
#print_time
#$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --n-iter 50 $ARGS_Linf
#print_time
#
#echo "-- MI-F(S)GM full - momentum 0.9, 50 iterations, ensemble all models, no random init --"
#MOMENTUM=0.9
#$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --momentum $MOMENTUM --n-iter 50 $ARGS_L2
#print_time
#$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --momentum $MOMENTUM --n-iter 50 $ARGS_Linf
#print_time

echo "-- PGD full - 5 random init, 50 iterations, ensemble all models, no random init --"
$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --n-random-init 5 --n-iter 50 $ARGS_L2
print_time  # estimated 8h30
$ATTACK $PATH_BNN $ARGS_COMMON $ARGS_FULL_ITER --n-random-init 5 --n-iter 50 $ARGS_Linf
print_time


# DNNs
for NB_DNN in 1 2 5 15 ; do
  echo "******** ENSEMBLE of $NB_DNN DNNs ********"
  PATH_DNNS="models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed50 --n-models-cycle 1 --limit-n-cycles $NB_DNN"

#  echo "-- I-F(S)GM light - 1K examples, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
#  $ATTACK $PATH_DNNS $ARGS_COMMON --n-iter 50 $ARGS_L2
#  print_time
#  $ATTACK $PATH_DNNS $ARGS_COMMON --n-iter 50 $ARGS_Linf
#  print_time
#
#  echo "-- MI-F(S)GM light - momentum 0.9, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
#  MOMENTUM=0.9
#  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_L2
#  print_time
#  $ATTACK $PATH_DNNS $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_Linf
#  print_time
#
#  echo "-- PGD light - 5 random init, 50 iterations, no ensemble (L2 and Linf norms) --"
#  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_L2
#  print_time
#  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_Linf
#  print_time

  ARGS_FULL_ITER="--n-ensemble $NB_DNN --batch-size 100"

#  ARGS_L2_FSGM="--norm 2 --max-norm 0.5 --norm-step 0.5"
#  ARGS_Linf_FSGM="--norm inf --max-norm 0.01568 --norm-step 0.01568"
#  echo "-- F(S)GM - 1 iteration, ensemble all models, no random init --"
#  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_L2_FSGM
#  print_time
#  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_Linf_FSGM
#  print_time
#
#  echo "-- I-F(S)GM full - 1K examples, 200 iterations, ensemble all models, no random init --"
#  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 50 $ARGS_L2
#  print_time
#  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 50 $ARGS_Linf
#  print_time
#
#  echo "-- MI-F(S)GM full - momentum 0.9, 50 iterations, ensemble all models, no random init --"
#  MOMENTUM=0.9
#  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --momentum $MOMENTUM --n-iter 50 $ARGS_L2
#  print_time
#  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --momentum $MOMENTUM --n-iter 50 $ARGS_Linf
#  print_time

  echo "-- PGD full - 5 random init, 50 iterations, ensemble all models --"
  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-random-init 5 --n-iter 50 $ARGS_L2
  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-random-init 5 --n-iter 50 $ARGS_Linf
  print_time

done
