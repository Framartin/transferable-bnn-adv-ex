#!/bin/bash -l
#SBATCH --time=0-04:00:00 # 4 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J EnsPgdHp      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 2 tasks
#SBATCH -c 4              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_attack_csgld_pgd__hp_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

ATTACK="python -u attack_csgld_pgd.py ./models/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1"

ARGS="--n-examples 1000 --norm 2 --max-norm 0.5 --norm-step 0.05 --seed 42 "
#ARGS="--n-examples 1000 --norm inf --max-norm 0.031 "

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0
SEED=0
echo "-- ensPGD - 150 iterations, no ensemble --"
$ATTACK --n-iter 150 --n-ensemble 1 --n-random-init 1 --iters-metrics 10 --shuffle $ARGS
print_time

echo "-- ensPGD - 150 iterations, 2 ensemble --"
$ATTACK --n-iter 150 --n-ensemble 2 --n-random-init 1 --iters-metrics 10 --shuffle $ARGS
print_time

echo "-- ensPGD - 150 iterations, 4 ensemble --"
$ATTACK --n-iter 150 --n-ensemble 4 --batch-size 32  --n-random-init 1 --iters-metrics 10 --shuffle $ARGS
print_time

echo "-- ensPGD - 15 iterations, 1 ensemble --"
$ATTACK --n-iter 15 --n-ensemble 1 --n-random-init 1 --iters-metrics 3 --shuffle $ARGS
print_time
echo "-- ensPGD - 15 iterations, 4 ensemble --"
$ATTACK --n-iter 15 --n-ensemble 4 --batch-size 32  --n-random-init 1 --iters-metrics 3 --shuffle $ARGS
print_time

echo "-- ensPGD - 15 iterations, 1 ensemble, 5 restarts --"
$ATTACK --n-iter 15 --n-ensemble 1 --n-random-init 5 --iters-metrics 3 --shuffle $ARGS
print_time
echo "-- ensPGD - 15 iterations, 4 ensemble, 5 restarts --"
$ATTACK --n-iter 15 --n-ensemble 4 --batch-size 32  --n-random-init 5 --iters-metrics 3 --shuffle $ARGS
print_time

