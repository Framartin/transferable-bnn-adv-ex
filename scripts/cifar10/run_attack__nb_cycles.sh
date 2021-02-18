#!/bin/bash -l
# bash scripts/cifar10/run_attack__nb_cycles.sh >>log/cifar10/run_attack__nb_cycles.log 2>&1

echo "\n CIFAR10 - HP: nb cycles \n"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=2

PATH_CSV="X_adv/CIFAR10/RQ/results_nb_cycles.csv"
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

for NB_CYCLES in {1..16} ; do
  echo "******** Limit to $NB_CYCLES cycles ********"
  PATH_BNN="models/CIFAR10/PreResNet110/cSGLD_cycles20_savespercycle3_it1 --n-models-cycle 3 --limit-n-cycles $NB_CYCLES"

  echo "-- I-F(S)GM light - 50 iterations, no ensemble (L2 and Linf norms), no random init --"
  $ATTACK $PATH_BNN $ARGS_COMMON --n-iter 50 $ARGS_L2
  print_time
  $ATTACK $PATH_BNN $ARGS_COMMON --n-iter 50 $ARGS_Linf
  print_time

  echo "-- MI-F(S)GM light - momentum 0.9, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
  MOMENTUM=0.9
  $ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_L2
  print_time
  $ATTACK $PATH_BNN $ARGS_COMMON --momentum $MOMENTUM --n-iter 50 $ARGS_Linf
  print_time

done
