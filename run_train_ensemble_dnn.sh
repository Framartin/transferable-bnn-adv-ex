#!/bin/bash -l
#SBATCH --time=3-00:00:00 # 3 days
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J TrainDNN       # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 2 tasks
#SBATCH -c 4              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_ensemble_dnn_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

ARGS='--num-workers 4'
print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
  SEED=$(($SEED+1))
}

SECONDS=0
SEED=1000  # different than target dnn (run_train_target_models.sh) and single dnn (run_train_dnn.sh)

# 15 independently train DNNs to compared with 15 cycles of cSGLD
python -u train.py ./models CIFAR10 PreResNet110 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --models 15 --seed $SEED $ARGS
print_time
# each takes 205 minutes to train

# train 3 DNNs for each architecture (same #epochs than cSGLD with 12x15 samples)
python -u train.py ./models CIFAR10 PreResNet110 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --models 4 --seed $SEED $ARGS
print_time # 3x 205 minutes

python -u train.py ./models CIFAR10 PreResNet164 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --models 3 --seed $SEED $ARGS
print_time # 3x 325 minutes

python -u train.py ./models CIFAR10 VGG16BN Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --models 3 --seed $SEED $ARGS
print_time # 3x60 min

python -u train.py ./models CIFAR10 VGG19BN Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --models 3 --seed $SEED $ARGS
print_time # 3x68 min

python -u train.py ./models CIFAR10 WideResNet28x10 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --models 3 --seed $SEED $ARGS
print_time # 3x 460 minutes
