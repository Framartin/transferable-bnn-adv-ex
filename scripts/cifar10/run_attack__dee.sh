#!/bin/bash -l
# bash scripts/cifar10/run_attack__dee.sh >>log/cifar10/run_attack__dee.log 2>&1

echo "\n CIFAR10 - RQ: nb of DNNs to have same transferability than cSGLD \n"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

PATH_CSV="X_adv/CIFAR10/PreResNet110/results_dee.csv"
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

# DNNs
for NB_DNN in {1..15} ; do
  echo "******** ENSEMBLE of $NB_DNN DNNs ********"
  PATH_DNNS="models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed50 --n-models-cycle 1 --limit-n-cycles $NB_DNN"

  echo "-- I-F(S)GM light - 1K examples, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
  # L2 I-FGM : deactivated because more than 15 (but used for graphs outside of DEE)
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

  ARGS_FULL_ITER="--n-ensemble $NB_DNN --batch-size 100"

  ARGS_L2_FSGM="--norm 2 --max-norm 0.5 --norm-step 0.5"
  ARGS_Linf_FSGM="--norm inf --max-norm 0.01568 --norm-step 0.01568"
  echo "-- F(S)GM - 1 iteration, ensemble all models, no random init --"
  # L2 FGM : deactivated because more than 15
  #  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_L2_FSGM
  #  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON $ARGS_FULL_ITER --n-iter 1 $ARGS_Linf_FSGM
  print_time

done

exit

# only run PGD for some values
echo "******* PGD light - 5 random init, 50 iterations, no ensemble (L2 and Linf norms) *******"
#for NB_DNN in {3..5} ; do
for NB_DNN in 3 ; do
  echo "-- Ensemble of $NB_DNN DNNs --"
  PATH_DNNS="models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed50 --n-models-cycle 1 --limit-n-cycles $NB_DNN"
  # L2 FGM : deactivated because more than 15
#  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_L2
#  print_time
  $ATTACK $PATH_DNNS $ARGS_COMMON --n-random-init 5 --n-iter 50 $ARGS_Linf
  print_time

done


