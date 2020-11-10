#!/bin/bash -l
#SBATCH --time=0-06:00:00 # 6 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J EnsPgdAr      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_attack_csgld_pgd__same_arch_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

ATTACK="python -u attack_csgld_pgd.py ./models/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1"

# computed on all data
ARGS_L2="--norm 2 --max-norm 0.5 --norm-step 0.05 --seed 42"
ARGS_Linf="--norm inf --max-norm 0.03137 --norm-step 0.003137 --seed 42"

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0

echo "-- ensPGD - 180 iterations, no ensemble, only 1K examples, with progress metrics (L2 and Linf norms) --"
$ATTACK --n-examples 1000 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle $ARGS_L2
print_time  # 18 minutes and 42 seconds
$ATTACK --n-examples 1000 --n-iter 180 --n-random-init 1 --iters-metrics 10 --shuffle $ARGS_Linf
print_time

echo "-- ensPGD - 180 iterations, no ensemble, all test set (L2 and Linf norms) --"
$ATTACK --skip-accuracy-computation --n-iter 180 --n-random-init 1 --iters-metrics 180 --shuffle $ARGS_L2
print_time # 31 minutes and 22 seconds
$ATTACK --skip-accuracy-computation --n-iter 180 --n-random-init 1 --iters-metrics 180 --shuffle $ARGS_Linf
print_time

