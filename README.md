# Transferable Adversarial Examples from BNN

Implementation of the paper **[Efficient and Transferable Adversarial Examples from Bayesian Neural Networks](https://gubri.eu/publication/transferable_adv_ex_from_bnn/)** by Martin Gubri, Maxime Cordy, Mike Papadakis and Yves Le Traon from the University of Luxembourg.

## Abstract

An established way to improve the transferability of black-box evasion attacks is to craft the adversarial examples on a surrogate ensemble model. Unfortunately, such methods involve heavy computation costs to train the models forming the ensemble. Based on a state-of-the-art Bayesian Neural Network technique, we propose a new method to efficiently build such surrogates by sampling from the posterior distribution of neural network weights during a single training process. Our experiments on ImageNet and CIFAR-10 show that our approach improves the transfer rates of four state-of-the-art attacks significantly (between 2.5 and 44.4 percentage points), in both intra-architecture and inter-architecture cases. On ImageNet, our approach can reach 94% of transfer rate while reducing training time from 387 to 136 hours on our infrastructure, compared to an ensemble of independently trained DNNs. Furthermore, our approach can be combined with test-time techniques improving transferability, further increasing their effectiveness by up to 25.1 percentage points.

## Install

```shell script
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir log
```

## Train

### 1. CIFAR-10

#### 1.1. Train Bayesian NN with cSGLD

For every 5 architectures, train 1 SG-MCMC with cSGLD for 5 cycles of 50 epochs and 3 samples per cycle.

```shell script
cd pytorch_ensembles
bash train_sse_mcmc_hpc.sh
bash train_sse_mcmc_hpc_pre164.sh
bash train_sse_mcmc_hpc_res50.sh
bash train_sse_mcmc_hpc_vgg16.sh
bash train_sse_mcmc_hpc_vgg19.sh
bash train_sse_mcmc_hpc_wide2810.sh
cd ..
```

#### 1.2. Train single deterministic DNN with Adam

For every 5 architectures, train 1 single DNN for 250 epochs.

```shell script
bash scripts/cifar10/run_train_dnn.sh
```

See help of `train.py` for more information:
```shell script
python train.py --help
```

### 1.3. Train Deep Ensembles with Adam

Train independently:
1. An ensemble of 15 deterministic DNNs with PreResNet110 architecture (250 epochs each)
2. An ensemble of 4 deterministic DNNs for every 5 architectures architecture (250 epochs each model) 

```shell script
bash scripts/cifar10/run_train_ensemble_dnn.sh
# copy the model trained in 1.2. to be used as the 4th model of each ensemble
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

### 1.4. Train target single deterministic DNN with Adam

For every 5 architectures, train 1 single DNN to be used as target model. The code and hyperparameters are the same than for models train in section 1.2, except the random seeds.

```shell script
batch run_train_target_models.sh
```

### 2. ImageNet

#### 2.1. Train Bayesian NN with cSGLD on resnet-50

For RQ1, train 1 resnet-50 SG-MCMC with cSGLD for 5 cycles of 45 epochs and 3 samples per cycle.

```shell script
cd pytorch_ensembles
bash train_imagenet_csgld_resnet50.sh
```

#### 2.2. Train Bayesian NN with cSGLD on 5 archs

For RQ2, train every 4 other architectures SG-MCMC with cSGLD for 3 cycles of 50 epochs and 3 samples per cycle.

```shell script
cd pytorch_ensembles
bash train_imagenet_csgld_resnext.sh
bash train_imagenet_csgld_densenet.sh
bash train_imagenet_csgld_mnasnet.sh
bash train_imagenet_csgld_efficientnet.sh
# training can be resume if necessary (in case of nan loss):
# bash train_imagenet_csgld_efficientnet_resume.sh
# bash train_imagenet_csgld_efficientnet_resume1.sh
cd ..
```

### 2.3. Retrieve and Train Deep Ensembles

For RQ1, we retrieve the resnet-50 models trained independently by [pytorch-ensembles](https://github.com/bayesgroup/pytorch-ensembles).

```shell script
bash scripts/imagenet/download_pretrained.sh
```

For RQ2, we train independently 1 DNN for every 4 other architectures (135 epochs each model).

```shell script
bash scripts/imagenet/train_dnn_densenet.sh
bash scripts/imagenet/train_dnn_efficientnet.sh
bash scripts/imagenet/train_dnn_mnasnet.sh
bash scripts/imagenet/train_dnn_resnext.sh
# in RQ2, to match the nb of epochs of 3 cycles of cSGLD, we train every architecture for an additional 5 epochs
bash scripts/imagenet/dnn_130to135epochs/train_dnn_densenet.sh
bash scripts/imagenet/dnn_130to135epochs/train_dnn_efficientnet_resume.sh
bash scripts/imagenet/dnn_130to135epochs/train_dnn_mnasnet.sh
bash scripts/imagenet/dnn_130to135epochs/train_dnn_resnet50.sh
bash scripts/imagenet/dnn_130to135epochs/train_dnn_resnext.sh
```

## Attack

### 1. RQ1: Intra-architecture transferability

Compute L2 and Linf adversarial examples against the PreResNet110 cSGLD and DNNs ensembles using FG(S)M, I-FG(S)M, MI-FG(S)M and PGD (in both variants: using 1 model or every models at each iteration).

Results are exported in CSV in `X_adv/CIFAR10/PreResNet110/results_same_arch.csv` and `X_adv/ImageNet/resnet50/results_same_arch.csv`. 

```shell script
# CIFAR
bash scripts/cifar10/run_attack_same_arch.sh
# ImageNet
bash scripts/imagenet/run_attack_same_arch.sh
```

To compute the T-DEE, attack every ensemble from 1 DNN to 15 DNNs. 
Results are exported in CSV in `X_adv/CIFAR10/PreResNet110/results_dee.csv` and `X_adv/ImageNet/resnet50/results_dee.csv`. 

```shell script
# CIFAR
bash scripts/cifar10/run_attack__dee.sh
# ImageNet
bash scripts/imagenet/run_attack__dee.sh
```

T-DEE values are computed in `plot_metrics.py`.

### 2. RQ2: Inter-architecture transferability

Compute L2 and Linf I-FG(S)M for each 5 hold-out architecture. On CIFAR-10, attack the following surrogate: cSGLD (15*4 models), 1 DNN (1*4 models) and ensemble of 4 DNNs per architecture (4*4 models). On ImageNet, attack the following surrogate: cSGLD (15*4 models) and 1 DNN per architecture (1*4 models). 

Results are exported in CSV in `X_adv/CIFAR10/holdout/results_holdout.csv` and `X_adv/ImageNet/holdout/results_holdout.csv`. 


```shell script
# CIFAR
bash scripts/cifar10/run_attack_multiple_archs.sh
# ImageNet cSGLD
bash scripts/imagenet/run_attack_csgld_pgd__multiple_archs.sh
# ImageNet DNNs
bash scripts/imagenet/run_attack_dnn__multiple_archs.sh
```

### 3. RQ3: test-time transferability techniques

Evaluate L2 and Linf I-FG(S)M with 3 test-time transferability techniques on both cSGLD and an ensemble of DNNs (1 for CIFAR, 2 for ImageNet).

Results are exported in CSV in `X_adv/cifar10/test_techniques/results_test_techniques.csv` and `X_adv/ImageNet/test_techniques/results_test_techniques.csv`.

````shell script
# CIFAR
bash scripts/cifar10/run_attack_with_test_techniques.sh
# ImageNet 
bash scripts/imagenet/run_attack_with_test_techniques.sh
````


### 4. RQ4: Hyperparameters

Train 1 cSGLD for 20 cycles.

````shell script
bash pytorch_ensembles/train_sse_mcmc_rq_nb_cycles.sh
````

Evaluate L2 and Linf I-FG(S)M on cSGLD with a number of cycles from 1 to 16 (on CIFAR-10 only).
Results are exported in CSV in `X_adv/CIFAR10/RQ/results_nb_cycles.csv`.

````shell script
bash scripts/cifar10/run_attack__nb_cycles.sh
````

## Attribution

cSGLD models are trained thanks to the work of Ruqi Zhang et al., and Arsenii Ashukha et al. available on [GitHub](https://github.com/bayesgroup/pytorch-ensembles).


## Supplementary Materials

### Number of cSGLD samples per cycle

Train 1 cSGLD for every number of cycles from 1 to 10.

````shell script
bash pytorch_ensembles/train_sse_mcmc_rq_nb_samples.sh
````

Evaluate L2 and Linf I-FG(S)M on cSGLD with a number of cycles from 1 to 16 (on CIFAR-10 only).
Results are exported in CSV in `X_adv/CIFAR10/RQ/results_nb_samples_per_cycle_true.csv`.

````shell script
bash scripts/cifar10/run_attack__nb_samples_true.sh
````

### Number attack iterations

Report transfer rates on cSGLD, 1, 2, 5 and 15 DNNs for every iteration up to 200. Computations are done for L2 and Linf norms of I-FG(S)M, MI-FG(S)M and PGD on both datasets.

Results are exported in CSV in `X_adv/CIFAR10/RQ/results_nb_iters.csv` and `X_adv/ImageNet/RQ/results_nb_iters.csv`.

````shell script
bash scripts/cifar10/run_rq_nb_iters.sh
bash scripts/imagenet/run_rq_nb_iters.sh
````

### 0-Gradient issue

Analyse the proportion of gradients smaller than the eps tolerance used in ART:

```shell script
python analyse_grad.py  # for CIFAR-10
python analyse_grad_imagenet.py  # for ImageNet
# modify line 11-12 to target a DNN or a cSGLD sample
```


## Miscellaneous

### Figure and Tables

Figure and Tables can be reproduced with the `plot_metrics.py` script.
