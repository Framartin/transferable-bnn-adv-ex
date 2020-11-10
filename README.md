# Transferable Adversarial Examples from BNN

Implementation of the paper **[Efficient and Transferable Adversarial Examples from Bayesian Neural Networks](https://gubri.eu/publication/transferable_adv_ex_from_bnn/)** by Martin Gubri, Maxime Cordy, Mike Papadakis and Yves Le Traon from the University of Luxembourg.

## Abstract

Deep neural networks are vulnerable to evasion attacks, i.e., carefully crafted examples designed to fool a model at test time. Attacks that successfully evade an ensemble of models can transfer to other independently trained models, which proves useful in black-box settings. Unfortunately, these methods involve heavy computation costs to train the models forming the ensemble. To overcome this, we propose a new method to generate transferable adversarial examples efficiently. Inspired by Bayesian deep learning, our method builds such ensembles by sampling from the posterior distribution of neural network weights during a single training process. Experiments on CIFAR-10 show that our approach improves the transfer rates significantly at equal or even lower computation costs. Intra-architecture transfer rate is increased by 23% compared to classical ensemble-based attacks, while requiring 4 times less training epochs. In the inter-architecture case, we show that we can combine our method with ensemble-based attacks to increase their transfer rate by up to 15% with constant training computational cost.

## Install

```shell script
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir log
```

## Train

The following code is designed to be run on a HPC with slurm to manage jobs. If `sbatch` is not available, replace it by `bash` in the following commands.


### 1. Train Bayesian DNN with cSGLD

For every 5 architectures, train 1 SG-MCMC with cSGLD for 15 cycles of 62 epochs and 12 samples per cycle.

```shell script
cd pytorch_ensembles
sbatch train_sse_mcmc_hpc.sh
sbatch train_sse_mcmc_hpc_pre164.sh
sbatch train_sse_mcmc_hpc_res50.sh
sbatch train_sse_mcmc_hpc_vgg16.sh
sbatch train_sse_mcmc_hpc_vgg19.sh
sbatch train_sse_mcmc_hpc_wide2810.sh
```

### 2. Train single deterministic DNN with Adam

For every 5 architectures, train 1 single DNN for 250 epochs.

```shell script
sbatch run_train_dnn.sh
```

See help of `train.py` for more information:
```shell script
python train.py --help
```

### 3. Train Deep Ensembles with Adam

Train independently:
1. An ensemble of 15 deterministic DNNs with PreResNet110 architecture (250 epochs each)
2. An ensemble of 4 deterministic DNNs for every 5 architectures architecture (250 epochs each model)

```shell script
sbatch run_train_ensemble_dnn
# copy the model trained in 2. to be used as the 4th model of each ensemble
cd models/CIFAR10/PreResNet110/
cp single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed100.pth dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1001/
cd PreResNet164
cp single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed103.pth dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1002/
cd ../VGG16BN/
cp single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed101.pth dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1003/
cd ../VGG19BN/
cp single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed102.pth dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1004/
cd ../WideResNet28x10/
cp single_model/model_Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed104.pth dnn_ensemble/Adam_bs128_lr0.01_lrd75_psig100.0_ep250_seed1005/
```

### 4. Train target single deterministic DNN with Adam

For every 5 architectures, train 1 single DNN to be used as target model. The code and hyperparameters are the same than for models train in section 2, except the random seeds.

```shell script
sbatch run_train_target.sh
```

## Attack

See help of `attack_csgld_pgd.py` for more information:
```shell script
python attack_csgld_pgd.py --help
```

### 0. Preliminary experiments

To run the variants implemented in `attack_csgld.py` that experiment different ways to iterate on samples (w/wo cycle structure,
w/wo shuffle, w/wo random init, several iteration per model vs. only 1, etc.)

*Can be safely skipped:*

```shell script
sbatch run_attack_csgld.sh
sbatch run_attack_csgld_pgd__hp.sh
```

### 1. Compute adversarial examples against cSGLD

Compute:
1. L2 and Linf adversarial examples against the PreResNet110 cSGLD
2. for each hold-out architecture, L2 adversarial examples against all but one architectures cSGLD ensembles


```shell script
sbatch run_attack_csgld_pgd__same_arch.sh
sbatch run_attack_csgld_pgd__multiple_archs.sh
```

Evaluate those adversarial examples against targeted DNNs

```shell script
sbatch run_evaluation_ensPGD_against_target.sh
```

### 2. Compute adversarial examples against single DNNs

Compute:
1. L2 and Linf adversarial examples against the PreResNet110 DNN
2. for each hold-out architecture, L2 adversarial examples against all but one architectures

```shell script
sbatch run_attack_ens_pgd__dnn.sh
```

Evaluate those adversarial examples against targeted DNNs

```shell script
sbatch run_evaluation_ensPGD_against_target__dnn.sh
```

### 3. Compute adversarial examples against ensembles of DNNs

Compute:
1. L2 and Linf adversarial examples against the ensemble of 15 PreResNet110 DNNs
2. for each hold-out architecture, L2 adversarial examples against all the ensembles of 4 DNNs of all but one architectures (ensemble of 4*4 models)

````shell script
sbatch run_attack_ens_pgd__dnn_ensemble.sh
````

Evaluate those adversarial examples against targeted DNNs

```shell script
sbatch run_evaluation_ensPGD_against_target__dnn_ensemble.sh
```

## Hyperparameters Analysis

### 1. Number of cycles

```shell script
sbatch run_attack_csgld_pgd__nb_cycles_10K.sh
# to craft only on 1K test examples for debug:
# sbatch run_attack_csgld_pgd__nb_cycles.sh
```

### 2. Number of samples per cycle

```shell script
sbatch run_attack_csgld_pgd__nb_samples_per_cycle_10K.sh
# to craft only on 1K test examples for debug:
# sbatch run_attack_csgld_pgd__nb_samples_per_cycle.sh
```


## Attribution

cSGLD models are trained thanks to the work of Ruqi Zhang et al., and Arsenii Ashukha et al. available on [GitHub](https://github.com/bayesgroup/pytorch-ensembles).
