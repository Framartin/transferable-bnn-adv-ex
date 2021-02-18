#!/bin/bash -l
#SBATCH --time=0-03:00:00 # 3 hours
#SBATCH --partition=gpu   # Use the batch partition reserved for passive jobs
#SBATCH -J AttTestTechs   # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/cifar10/run_attack_with_test_techniques_%j.log"
#SBATCH --mail-type=all
#command -v module >/dev/null 2>&1 && module load lang/Python
#source venv/bin/activate


echo "run RQ with/without test time techniques on DNN vs on cSGLD"

# launch with:
# bash scripts/cifar10/run_attack_with_test_techniques.sh >>log/cifar10/run_attack_with_test_techniques.log 2>&1

source /opt/miniconda/bin/activate
conda activate pytorch
export CUDA_VISIBLE_DEVICES=3

set -x

ATTACK="python -u attack_csgld_pgd_torch.py"


PATH_CSV="X_adv/cifar10/test_techniques/results_test_techniques.csv"

CSGLD_MODELS="models/CIFAR10/PreResNet110/cSGLD_cycles5_savespercycle3_it1 --shuffle"
DNN_MODELS="models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed50 --n-models-cycle 1 --limit-n-cycles 1"  # limit 1 DNN
TARGET="models_target/CIFAR10/PreResNet110/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1.pth"

N_ITERS=50


# args
ARGS_L2="--norm 2 --max-norm 0.5 --norm-step 0.05"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
ARGS_COMMON="--model-target-path $TARGET --n-iter $N_ITERS --seed 42 --no-save --csv-export ${PATH_CSV}"

TEST_TECHS_ARGS_LIST=(
  "--ghost-attack"           # ghost
  "--input-diversity"        # input diversity
  "--translation-invariant"  # translation invariance
  ""                         # baseline
)

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0


for TEST_TECHS_ARG in "${TEST_TECHS_ARGS_LIST[@]}" ; do
  echo "\n ------- Test time transfer attack: $TEST_TECHS_ARG ------- \n"
  echo "---- I-F(S)GM - 50 iterations, ensemble only 1 at each iter (L2 and Linf norms), no random init ----"

  echo "-- cSGLD --"
  # ~ train computational cost, same test computational cost
  $ATTACK $CSGLD_MODELS $TEST_TECHS_ARG $ARGS_L2 $ARGS_COMMON
  print_time
  $ATTACK $CSGLD_MODELS $TEST_TECHS_ARG $ARGS_Linf $ARGS_COMMON
  print_time

  echo "-- 1 DNN --"
  $ATTACK $DNN_MODELS $TEST_TECHS_ARG $ARGS_L2 $ARGS_COMMON
  print_time
  $ATTACK $DNN_MODELS $TEST_TECHS_ARG $ARGS_Linf $ARGS_COMMON
  print_time

done
