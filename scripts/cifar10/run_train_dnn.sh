#!/bin/bash -l
#SBATCH --time=1-00:00:00 # 1 day
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH -J TrainDNN       # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_dnn_%j.log"
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
SEED=100  # different than target dnn

# similar config than https://github.com/BIGBALLON/CIFAR-ZOO/blob/master/experiments/cifar10/preresnet110/config.yaml
python -u train.py ./models CIFAR10 PreResNet110 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 216 minutes

# with LR on plateau, same #epochs budget
#python -u train.py ./models CIFAR10 PreResNet110 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
#print_time
# test accuracy low (89% vs. lr step decay of 92.8%)

python -u train.py ./models CIFAR10 VGG16BN Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 60 minutes

python -u train.py ./models CIFAR10 VGG19BN Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 68 minutes

python -u train.py ./models CIFAR10 PreResNet164 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 325 minutes

python -u train.py ./models CIFAR10 WideResNet28x10 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 460 minutes
