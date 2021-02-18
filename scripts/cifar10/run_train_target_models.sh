#!/bin/bash -l
#SBATCH --time=1-10:00:00 # 1 day 20 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH -J TrainTargets      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_target_%j.log"
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
SEED=10

# similar HP than https://github.com/BIGBALLON/CIFAR-ZOO/blob/master/experiments/cifar10/preresnet110/config.yaml
python -u train.py ./models_target CIFAR10 PreResNet110 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 216 minutes

python -u train.py ./models_target CIFAR10 PreResNet164 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 9h estimate

python -u train.py ./models_target CIFAR10 VGG16BN Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 60 min

python -u train.py ./models_target CIFAR10 VGG19BN Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 68 min

python -u train.py ./models_target CIFAR10 WideResNet28x10 Adam --lr 0.01 --lr-decay 75 --lr-decay-gamma 0.1 --prior-sigma 100 --epochs 250 --seed $SEED $ARGS
print_time # 9h estimate
