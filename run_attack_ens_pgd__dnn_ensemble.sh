#!/bin/bash -l
#SBATCH --time=0-20:00:00 # 20 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J Pgd4DNN      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_attack_ens_pgd__dnn_ensemble_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

ATTACK="python -u attack_csgld_pgd.py"


ARGS="--skip-accuracy-computation --norm 2 --max-norm 0.5 --norm-step 0.05 --seed 42"
#ARGS="--n-examples 1000 --norm inf --max-norm 0.031"

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

PRN110="./models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1001"
PRN164="./models/CIFAR10/PreResNet164/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1002"
VGG16BN="./models/CIFAR10/VGG16BN/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1003"
VGG19BN="./models/CIFAR10/VGG19BN/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1004"
WIDE2810="./models/CIFAR10/WideResNet28x10/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1005"

SECONDS=0

echo "----- Attack against 15 PreResNet110 DNNs for transferability to PreResNet110 (on all dataset) -----"

ARGS_L2="--norm 2 --max-norm 0.5 --norm-step 0.05 --seed 42 "
ARGS_Linf="--norm inf --max-norm 0.03137 --norm-step 0.003137 --seed 42 "
echo "-- ensPGD - 180 iterations, no ensemble, all test set (L2 and Linf norms) --"
PRN110_15DNN="./models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed50"
$ATTACK $PRN110_15DNN --skip-accuracy-computation --n-iter 180 --n-random-init 1 --iters-metrics 180 --shuffle $ARGS_L2
print_time # 31 minutes and 22 seconds
$ATTACK $PRN110_15DNN --skip-accuracy-computation --n-iter 180 --n-random-init 1 --iters-metrics 180 --shuffle $ARGS_Linf
print_time


echo "----- Attack against 4 DNNs for each architecture -----"
# last 4th copied from models/single_model  (the one used in single DNN attack)

echo "-- ensPGD - 180 iterations, PreResNet164 VGG16BN VGG19BN WIDE2810 (PreResNet110 holdout) --"
$ATTACK $PRN164 $VGG16BN $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time  # 153 minutes

echo "-- ensPGD - 180 iterations, PreResNet110 VGG16BN VGG19BN WIDE2810 (PreResNet164 holdout) --"
$ATTACK $PRN110 $VGG16BN $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time # 138 minutes

echo "-- ensPGD - 180 iterations, PreResNet110 PreResNet164 VGG19BN WIDE2810 (VGG16BN holdout) --"
$ATTACK $PRN110 $PRN164 $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time # 178 min

echo "-- ensPGD - 180 iterations, PreResNet110 PreResNet164 VGG16BN WIDE2810 (VGG19BN holdout) --"
$ATTACK $PRN110 $PRN164 $VGG16BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time

echo "-- ensPGD - 180 iterations, PreResNet110 PreResNet164 VGG16BN VGG19BN (WIDE2810 holdout) --"
$ATTACK $PRN110 $PRN164 $VGG16BN $VGG19BN --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time

echo "-- ensPGD - 180 iterations, ALL: PreResNet110 PreResNet164 VGG16BN VGG19BN WIDE2810 --"
$ATTACK $PRN110 $PRN164 $VGG16BN $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time
