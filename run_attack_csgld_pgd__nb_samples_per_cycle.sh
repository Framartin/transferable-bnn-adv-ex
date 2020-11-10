#!/bin/bash -l
#SBATCH --time=0-02:00:00 # 2 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J NbSplPC       # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 4              # 4 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_attack_csgld_pgd__nb_samples_per_cycle_%j.log"
#SBATCH --mail-type=end,fail


command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate

set -x

# create a sim link to easily distinguish all the X_adv generating iteratively
REL_PATH_MODEL="./models/CIFAR10/PreResNet110/"
MODEL="cSGLD_cycles15_savespercycle12_it1"
MODEL_XP="${MODEL}__nbsamplespercycle"
cd $REL_PATH_MODEL || exit
ln -s "$MODEL" "$MODEL_XP"
cd ../../.. || exit

ARGS='--n-examples 1000 --norm 2 --max-norm 0.5 --norm-step 0.125 --seed 42'

print_time() {
  duration=$SECONDS
  echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
  SECONDS=0
}

SECONDS=0
echo "\n--- PGD 100 iters applied on different numbers of samples per cycle ranging from 1 to 12 cycles; limit on first 4 cycles."
for i in $(seq 1 12); do
  echo "Limit $i samples per cycle (at regular interval - sampling rate)"
  python -u attack_csgld_pgd.py "${REL_PATH_MODEL}${MODEL_XP}" --limit-n-samples-per-cycle $i --limit-n-cycles 4 --n-models-cycle 12 --n-iter 100 --n-random-init 1 --iters-metrics 100 --shuffle $ARGS
  print_time
  # 4 minutes 30 sec per call
  echo "Limit $i samples per cycle (last ones)"
  python -u attack_csgld_pgd.py "${REL_PATH_MODEL}${MODEL_XP}" --limit-n-samples-per-cycle $i --method-samples-per-cycle last --limit-n-cycles 4 --n-models-cycle 12 --n-iter 100 --n-random-init 1 --iters-metrics 100 --shuffle $ARGS
  print_time
  echo "Limit $i samples per cycle (first ones)"
  python -u attack_csgld_pgd.py "${REL_PATH_MODEL}${MODEL_XP}" --limit-n-samples-per-cycle $i --method-samples-per-cycle first --limit-n-cycles 4 --n-models-cycle 12 --n-iter 100 --n-random-init 1 --iters-metrics 100 --shuffle $ARGS
  print_time
done

echo "-------------------------------------------"
echo "EVALUATION OF CRAFTED ADV EX"

PATH_ADVEX="${REL_PATH_MODEL/models/X_adv}${MODEL_XP}"
python evaluate_x_adv_against_target.py "models_target/CIFAR10/PreResNet110/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1.pth" --export_csv "${PATH_ADVEX}/eval_target.csv" --directory_x_adv "$PATH_ADVEX"
