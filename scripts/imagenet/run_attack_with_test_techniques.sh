#!/bin/bash -l

echo "run RQ with/without test time techniques on DNN vs on cSGLD"

# launch with:
# bash scripts/imagenet/run_attack_with_test_techniques.sh >>log/imagenet/run_attack_with_test_techniques.log 2>&1

source /opt/miniconda/bin/activate
conda activate pytorch

export CUDA_VISIBLE_DEVICES=3

set -x

ATTACK="python -u attack_csgld_pgd_torch.py"


PATH_CSV="X_adv/ImageNet/test_techniques/results_test_techniques.csv"
DATAPATH="../data/ILSVRC2012"

CSGLD_MODELS="./models/ImageNet/resnet50/cSGLD_cycles5_samples3_bs256 --shuffle"
DNN_MODELS="./models/ImageNet/resnet50/deepens_imagenet --n-models-cycle 1"  # limit 2 DNNs
TARGET="ImageNet/pretrained/resnet50"

N_ITERS=50
N_EXAMPLES=5000

# args
ARGS_L2="--norm 2 --max-norm 3 --norm-step 0.3"
ARGS_Linf="--norm inf --max-norm 0.01568 --norm-step 0.001568"
ARGS_COMMON="--model-target-path $TARGET --n-examples $N_EXAMPLES --n-iter $N_ITERS --data-path $DATAPATH --batch-size 32 --seed 42 --no-save --csv-export ${PATH_CSV}"

TEST_TECHS_ARGS_LIST=(
  ""                         # baseline
  "--ghost-attack"           # ghost
  "--input-diversity"        # input diversity
  "--translation-invariant"  # translation invariance
)

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0


for TEST_TECHS_ARG in "${TEST_TECHS_ARGS_LIST[@]}" ; do
  echo "\n ------- Test time transfer attack: $TEST_TECHS_ARG ------- \n"
  echo "---- I-F(S)GM - 5K examples, 50 iterations, ensemble only 2 at each iter (L2 and Linf norms), no random init ----"

  echo "-- cSGLD --"
  # ~ train computational cost, same test computational cost
  $ATTACK $CSGLD_MODELS $TEST_TECHS_ARG --n-ensemble 2 $ARGS_L2 $ARGS_COMMON
  print_time # 37 minutes 49 secs
  $ATTACK $CSGLD_MODELS $TEST_TECHS_ARG --n-ensemble 2 $ARGS_Linf $ARGS_COMMON
  print_time

  echo "-- 2 DNNs --"
  $ATTACK $DNN_MODELS $TEST_TECHS_ARG --limit-n-cycles 2 --n-ensemble 2 $ARGS_L2 $ARGS_COMMON
  print_time
  $ATTACK $DNN_MODELS $TEST_TECHS_ARG --limit-n-cycles 2 --n-ensemble 2 $ARGS_Linf $ARGS_COMMON
  print_time

done
