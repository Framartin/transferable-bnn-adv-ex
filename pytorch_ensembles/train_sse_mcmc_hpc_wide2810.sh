#!/bin/bash -l
#SBATCH --time=0-30:00:00 # 30 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J TrainWide2810  # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_csgld_wide2810_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python

source ../venv/bin/activate
set -x

bash ./train_sse_mcmc.sh CIFAR10 WideResNet28x10 1 ../models ../data cSGLD
