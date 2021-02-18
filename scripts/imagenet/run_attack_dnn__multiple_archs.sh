#!/bin/bash -l
#SBATCH --time=0-20:00:00 # 9 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH -J EnsPgdArchs      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/imagenet/run_attack_dnn__multiple_archs_%j.log"
#SBATCH --mail-type=end,fail


#command -v module >/dev/null 2>&1 && module load lang/Python
#source venv/bin/activate

# launch with:
# bash scripts/imagenet/run_attack_dnn__multiple_archs.sh >>log/imagenet/run_attack_dnn__multiple_archs.log 2>&1
source /opt/miniconda/bin/activate
conda activate pytorch

export CUDA_VISIBLE_DEVICES=3

set -x

ATTACK="python -u attack_csgld_pgd_torch.py"


PATH_CSV="X_adv/ImageNet/holdout/results_holdout.csv"
DATAPATH="../data/ILSVRC2012"

# restrict number to 1 model (resnet-50 has 50 models available)
# seed to ensure to have the same test subset
ARGS_common="--limit-n-cycles 1 --n-models-cycle 1 --seed 42 --no-save --csv-export ${PATH_CSV}"
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3 $ARGS_common"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568 $ARGS_common"

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

# paths
MODELS=(
    "./models/ImageNet/resnet50/single_model/"
    "./models/ImageNet/resnext50_32x4d/single_model/"
    "./models/ImageNet/densenet121/single_model/"
    "./models/ImageNet/mnasnet1_0/single_model/"
    "./models/ImageNet/efficientnet_b0/single_model/"
)

SECONDS=0

# for each arch in holdout, compute adv ex on the other archs

NB_MODELS=${#MODELS[@]}
for MODEL in ${MODELS[@]} ; do
  # extract arch name
  TMP=(`echo $MODEL | tr '/' ' '`)
  NAME=${TMP[3]}
  echo "\n ---- Arch holdout: $NAME ---- \n"
  MODELS_KEEP=( "${MODELS[@]/$MODEL}" )
  TARGET="ImageNet/pretrained/${NAME}"
  # attack
  echo "-- I-F(S)GM - 1K examples, 50 iterations, no ensemble (L2 and Linf norms), no random init --"
  N_ITERS=50
  N_EXAMPLES=5000
  $ATTACK ${MODELS_KEEP[@]} --data-path $DATAPATH --model-target-path $TARGET --n-examples $N_EXAMPLES --n-iter $N_ITERS --shuffle --batch-size 32 $ARGS_L2
  print_time
  $ATTACK ${MODELS_KEEP[@]} --data-path $DATAPATH --model-target-path $TARGET --n-examples $N_EXAMPLES --n-iter $N_ITERS --shuffle --batch-size 32 $ARGS_Linf
  print_time
done
