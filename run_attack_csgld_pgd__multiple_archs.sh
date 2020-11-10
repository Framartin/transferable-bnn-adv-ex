#!/bin/bash -l
#SBATCH --time=0-20:00:00 # 9 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J EnsPgdArchs      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_attack_csgld_pgd__multiple_archs_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

ATTACK="python -u attack_csgld_pgd.py"

ARGS="--skip-accuracy-computation --norm 2 --max-norm 0.5 --norm-step 0.05 --seed 42 "
#ARGS="--n-examples 1000 --norm inf --max-norm 0.031 "

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

PRN110="./models/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1"
PRN164="./models/CIFAR10/PreResNet164/cSGLD_cycles15_savespercycle12_it1__178samples"
VGG16BN="./models/CIFAR10/VGG16BN/cSGLD_cycles15_savespercycle12_it1"
VGG19BN="./models/CIFAR10/VGG19BN/cSGLD_cycles15_savespercycle12_it1"
WIDE2810="./models/CIFAR10/WideResNet28x10/cSGLD_cycles15_savespercycle12_it1"

SECONDS=0

echo "-- ensPGD - 180 iterations, PreResNet164 VGG16BN VGG19BN WIDE2810 (PreResNet110 holdout) --"
$ATTACK $PRN164 $VGG16BN $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time  # 158 minutes

echo "-- ensPGD - 180 iterations, PreResNet110 VGG16BN VGG19BN WIDE2810 (PreResNet164 holdout) --"
$ATTACK $PRN110 $VGG16BN $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time # 143 minutes

echo "-- ensPGD - 180 iterations, PreResNet110 PreResNet164 VGG19BN WIDE2810 (VGG16BN holdout) --"
$ATTACK $PRN110 $PRN164 $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time

echo "-- ensPGD - 180 iterations, PreResNet110 PreResNet164 VGG16BN WIDE2810 (VGG19BN holdout) --"
$ATTACK $PRN110 $PRN164 $VGG16BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time

echo "-- ensPGD - 180 iterations, PreResNet110 PreResNet164 VGG16BN VGG19BN (WIDE2810 holdout) --"
$ATTACK $PRN110 $PRN164 $VGG16BN $VGG19BN --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time

echo "-- ensPGD - 180 iterations, ALL: PreResNet110 PreResNet164 VGG16BN VGG19BN WIDE2810 --"
$ATTACK $PRN110 $PRN164 $VGG16BN $VGG19BN $WIDE2810 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle --batch-size 64 $ARGS
print_time
