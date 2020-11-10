#!/bin/bash -l
#SBATCH --time=0-03:00:00 # 3 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J AttCSGLD      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 2              # 2 tasks
#SBATCH -c 2              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_attack_sgld_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

ATTACK="python -u attack_csgld.py ./models/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1 12"


ARGS="--n-examples 1000 --norm 2 --max-norm 0.5 "
#ARGS="--n-examples 1000 --norm inf --max-norm 0.031 "

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
  SEED=$(($SEED+1))
}

SECONDS=0
SEED=0
# 1 ensemble per cycle: lower batch-size to avoid out of memory
echo "-- 15 steps - 1 ensemble per cycle --"
$ATTACK FGM --ensemble-inner 12 --batch-size 32 --n-iter-outer 15 --norm-inner 0.125 --seed $SEED $ARGS
print_time # 19.5 min
# + random init
echo "-- 15 steps - 1 ensemble per cycle with random init --"
$ATTACK FGM --ensemble-inner 12 --batch-size 32 --n-iter-outer 15 --n-random-init-outer 1 --norm-inner 0.125 --seed $SEED $ARGS
print_time # 19.5 min

# only one cycle
echo "-- 15 steps - only one cycle (control) --"
$ATTACK FGM --ensemble-inner 1 --n-iter-inner 15 --n-iter-outer 1 --shuffle-outer --norm-inner 0.5 --seed $SEED $ARGS
print_time

# pure iteratively (in order)
echo "-- 15*12 steps - pure iteration on models --"
$ATTACK FGM --ensemble-inner 1 --n-iter-inner 12 --n-iter-outer 15 --norm-inner 0.125 --seed $SEED $ARGS
print_time

# pure iteratively (in random order)
echo "-- 15*12 steps - iteration on models in random order --"
$ATTACK FGM --ensemble-inner 1 --n-iter-inner 12 --n-iter-outer 15 --norm-inner 0.125 --shuffle-inner --shuffle-outer --seed $SEED $ARGS
print_time

# test time difference with ensemble
echo "-- 15*12 steps - ensemble of 3 samples --"
$ATTACK FGM --ensemble-inner 3 --n-iter-inner 12 --n-iter-outer 15 --norm-inner 0.125 --shuffle-inner --shuffle-outer --seed $SEED $ARGS
print_time # 22.5 minutes

# all samples in random order
# 3h long because it predict with the entire ensemble 180 times
echo "-- 180 steps - all samples in random order (no cycles) --"
python -u attack_csgld.py ./models/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1 1 FGM --ensemble-inner 1 --n-iter-inner 1 --n-iter-outer 180 --norm-inner 0.125 --shuffle-inner --shuffle-outer --seed $SEED $ARGS
print_time

# expensive
# 37 minutes
echo "-- 15*1 steps - expensive PGD with random init --"
$ATTACK PGD --n-iter-attack 10 --ensemble-inner 3 --n-iter-outer 15 --norm-inner 0.125 --n-random-init-outer 1 --n-random-init-inner 5 --shuffle-outer --seed $SEED $ARGS
print_time

echo "-- 15*4 steps - expensive PGD with random init and inner shuffle --"
$ATTACK PGD --n-iter-attack 10 --ensemble-inner 3 --n-iter-outer 15 --n-iter-inner 4 --norm-inner 0.125 --n-random-init-outer 1 --n-random-init-inner 5 --shuffle-inner --shuffle-outer --seed $SEED $ARGS
print_time # 113 minutes

# PGD without ensemble
echo "-- 15*12 steps - expensive PGD without inner ensembling --"
$ATTACK PGD --n-iter-attack 10 --ensemble-inner 1 --n-iter-outer 15 --n-iter-inner 12 --norm-inner 0.125 --n-random-init-outer 1 --n-random-init-inner 5 --shuffle-outer --seed $SEED $ARGS
print_time # 114 minutes
