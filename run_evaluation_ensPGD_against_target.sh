#!/bin/bash -l
#SBATCH --time=0-00:10:00 # 10 minutes
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J EvalXadv   # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 2 tasks
#SBATCH -c 4              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_evaluation_enspgd_%j.log"
#SBATCH --mail-type=end,fail


echo "Evaluate crafted cSGLD X_adv against a target DNN model (transfer attack)"

command -v module >/dev/null 2>&1 && module load lang/Python
source venv/bin/activate
set -x

TARGET_PRN110="models_target/CIFAR10/PreResNet110/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1.pth"
TARGET_PRN164="models_target/CIFAR10/PreResNet164/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed3.pth"
TARGET_VGG16BN="models_target/CIFAR10/VGG16BN/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed10.pth"
TARGET_VGG19BN="models_target/CIFAR10/VGG19BN/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed11.pth"
TARGET_WIDE2810="models_target/CIFAR10/WideResNet28x10/single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed12.pth"

echo "\n-----------------------------------------------------------------------------"
echo "********* Evaluate the variants in run_attack_csgld_pgd.py on same architecture (different seed, optimizer, number of epochs) *******"
python -u evaluate_x_adv_against_target.py $TARGET_PRN110 --directory_x_adv X_adv/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1 --export_csv X_adv/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1/eval_target.csv

echo "\n\n-----------------------------------------------------------------------------"
echo "********* Evaluate the multi-architectures against the 5 hold-out architectures in run_attack_csgld_pgd.py ********"


echo "\n----- Adv ex crafted against PreResNet164 VGG16BN VGG19BN WideResNet28x10 (PreResNet110 holdout) ----"
Xadv_DIR="X_adv/CIFAR10/multiPreResNet164_VGG16BN_VGG19BN_WideResNet28x10/cSGLD_cycles15_savespercycle12_it1__178samples/PGDens_ens1_niter180_nrestart1iter0_shuffleTrue/"
python -u evaluate_x_adv_against_target.py $TARGET_PRN110 --directory_x_adv $Xadv_DIR

echo "\n----- Adv ex crafted against PreResNet110 VGG16BN VGG19BN WideResNet28x10 (PreResNet164 holdout) ----"
Xadv_DIR="X_adv/CIFAR10/multiPreResNet110_VGG16BN_VGG19BN_WideResNet28x10/cSGLD_cycles15_savespercycle12_it1/PGDens_ens1_niter180_nrestart1iter0_shuffleTrue/"
python -u evaluate_x_adv_against_target.py $TARGET_PRN164 --directory_x_adv $Xadv_DIR

echo "\n----- Adv ex crafted against PreResNet110 PreResNet164 VGG19BN WideResNet28x10 (VGG16BN holdout) ----"
Xadv_DIR="X_adv/CIFAR10/multiPreResNet110_PreResNet164_VGG19BN_WideResNet28x10/cSGLD_cycles15_savespercycle12_it1/PGDens_ens1_niter180_nrestart1iter0_shuffleTrue/"
python -u evaluate_x_adv_against_target.py $TARGET_VGG16BN --directory_x_adv $Xadv_DIR

echo "\n----- Adv ex crafted against PreResNet110 PreResNet164 VGG16BN WideResNet28x10 (VGG19BN holdout) ----"
Xadv_DIR="X_adv/CIFAR10/multiPreResNet110_PreResNet164_VGG16BN_WideResNet28x10/cSGLD_cycles15_savespercycle12_it1/PGDens_ens1_niter180_nrestart1iter0_shuffleTrue/"
python -u evaluate_x_adv_against_target.py $TARGET_VGG19BN --directory_x_adv $Xadv_DIR

echo "\n----- Adv ex crafted against PreResNet110 PreResNet164 VGG16BN VGG19BN (WideResNet28x10 holdout) ----"
Xadv_DIR="X_adv/CIFAR10/multiPreResNet110_PreResNet164_VGG16BN_VGG19BN/cSGLD_cycles15_savespercycle12_it1/PGDens_ens1_niter180_nrestart1iter0_shuffleTrue/"
python -u evaluate_x_adv_against_target.py $TARGET_WIDE2810 --directory_x_adv $Xadv_DIR


echo "\n\n-----------------------------------------------------------------------------"
echo "********* Evaluate the multi-architectures against others hold-out architectures in run_attack_csgld_pgd.py ********"

echo "\n----- Adv ex crafted against PreResNet110 PreResNet164 VGG16BN VGG19BN WideResNet28x10 (all, no holdout) ----"
Xadv_DIR="X_adv/CIFAR10/multiPreResNet110_PreResNet164_VGG16BN_VGG19BN_WideResNet28x10/cSGLD_cycles15_savespercycle12_it1/PGDens_ens1_niter180_nrestart1iter0_shuffleTrue"
echo "-- /!\ Below are not hold-out --"
python -u evaluate_x_adv_against_target.py $TARGET_PRN110 --directory_x_adv $Xadv_DIR
python -u evaluate_x_adv_against_target.py $TARGET_PRN164 --directory_x_adv $Xadv_DIR
python -u evaluate_x_adv_against_target.py $TARGET_VGG16BN --directory_x_adv $Xadv_DIR
python -u evaluate_x_adv_against_target.py $TARGET_VGG19BN --directory_x_adv $Xadv_DIR
python -u evaluate_x_adv_against_target.py $TARGET_WIDE2810 --directory_x_adv $Xadv_DIR
