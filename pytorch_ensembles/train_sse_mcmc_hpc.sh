#!/bin/bash -l
#SBATCH --time=0-05:00:00 # 5 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH -J TrainCycl      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 2 tasks
#SBATCH -c 4              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_csgld_%j.log"
#SBATCH --mail-type=end,fail

command -v module >/dev/null 2>&1 && module load lang/Python

source ../venv/bin/activate
set -x


bash ./train_sse_mcmc.sh CIFAR10 PreResNet110 1 ../models ../data cSGLD
# on single GPU PreResNet110: 5 cycles of 50 epochs with 3 saves on 3.66 hours (220')
