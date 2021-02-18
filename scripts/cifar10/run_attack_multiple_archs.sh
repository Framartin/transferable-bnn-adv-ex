#!/bin/bash -l

# launch with:
# bash scripts/cifar10/run_attack_multiple_archs.sh >>log/cifar10/run_attack_multiple_archs.log 2>&1
source /opt/miniconda/bin/activate
conda activate pytorch

export CUDA_VISIBLE_DEVICES=3

set -x

ATTACK="python -u attack_csgld_pgd_torch.py"


PATH_CSV="X_adv/CIFAR10/holdout/results_holdout.csv"

# restrict number to 1 model (resnet-50 has 50 models available)
ARGS_common="--seed 42 --no-save --csv-export ${PATH_CSV}"
ARGS_L2="--norm 2 --max-norm 0.5 --norm-step 0.05 $ARGS_common"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568 $ARGS_common"


print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

# paths
CSGLD_MODELS=(
    "models/CIFAR10/PreResNet110/cSGLD_cycles5_savespercycle3_it1/"
    "models/CIFAR10/PreResNet164/cSGLD_cycles5_savespercycle3_it1/"
    "models/CIFAR10/VGG16BN/cSGLD_cycles5_savespercycle3_it1/"
    "models/CIFAR10/VGG19BN/cSGLD_cycles5_savespercycle3_it1/"
    "models/CIFAR10/WideResNet28x10/cSGLD_cycles5_savespercycle3_it1/"
)

SINGLE_DNN=(
    "models/CIFAR10/PreResNet110/single_model/"
    "models/CIFAR10/PreResNet164/single_model/"
    "models/CIFAR10/VGG16BN/single_model/"
    "models/CIFAR10/VGG19BN/single_model/"
    "models/CIFAR10/WideResNet28x10/single_model/"
)

ENS_DNNS=(
    "models/CIFAR10/PreResNet110/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1001/"
    "models/CIFAR10/PreResNet164/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1002/"
    "models/CIFAR10/VGG16BN/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1003/"
    "models/CIFAR10/VGG19BN/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1004/"
    "models/CIFAR10/WideResNet28x10/dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1005/"
)

MODELS_TARGET=(
    "models_target/CIFAR10/PreResNet110/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1.pth"
    "models_target/CIFAR10/PreResNet164/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed3.pth"
    "models_target/CIFAR10/VGG16BN/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed10.pth"
    "models_target/CIFAR10/VGG19BN/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed11.pth"
    "models_target/CIFAR10/WideResNet28x10/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed12.pth"
)

SECONDS=0

# array of arrays
SURROGATES_MODELS=("CSGLD_MODELS" "SINGLE_DNN" "ENS_DNNS")

for SURROGATES in "${SURROGATES_MODELS[@]}"; do
  MODELS="$SURROGATES[@]"
  i=0
  # for each arch in holdout, compute adv ex on the other archs
  for MODEL in "${!MODELS}"; do
    # extract arch name
    TARGET="${MODELS_TARGET[i]}"
    TMP=(`echo $MODEL | tr '/' ' '`)
    NAME=${TMP[2]}
    echo "\n ---- Arch holdout: $NAME ---- \n"
    MODELS_=( "${!MODELS}" )
    MODELS_KEEP=( "${MODELS_[@]/$MODEL}" )

    # attack
    echo "-- I-F(S)GM - 50 iterations, no ensemble (L2 and Linf norms), no random init --"
    N_ITERS=50
    $ATTACK ${MODELS_KEEP[@]} --model-target-path $TARGET --n-iter $N_ITERS $ARGS_L2
    print_time
    $ATTACK ${MODELS_KEEP[@]} --model-target-path $TARGET --n-iter $N_ITERS $ARGS_Linf
    print_time

    ((i=i+1))
  done
done
