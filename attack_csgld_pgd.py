"""
Implementation of the attack used in the article
"""

import numpy as np
import pandas as pd
import torch
import argparse
import time
import os
import random
from random import shuffle
from itertools import cycle
from torch import cuda
from art.attacks.evasion import FastGradientMethod
from art.utils import projection, random_sphere
from art.config import ART_NUMPY_DTYPE
from utils.data import CIFAR10, CIFAR100
from utils.helpers import guess_model, guess_and_load_model, load_classifier, load_classifier_ensemble, list_models, compute_accuracy_multiple_ensemble, save_numpy, compute_norm, USE_CUDA
import torch.backends.cudnn as cudnn


cudnn.benchmark = True
cudnn.deterministic = True

# parse args
parser = argparse.ArgumentParser(description="Craft PGD adv ex with each update computed on a different samples from an ensemble of models trained with cSGLD")
parser.add_argument("dirs_models", nargs='+', help="Path to directory containing all the models file of the ensemble model")
parser.add_argument('--n-iter', type=int, default=None, help="Number of iterations to perform. If None (default), set to the number of samples.")
parser.add_argument("--norm", choices=['1', '2', 'inf'], default='2', help="Type of L-norm to use. Default: 2")
parser.add_argument("--max-norm", type=float, required=True, help="Max L2 norm of the perturbation")
parser.add_argument("--norm-step", type=float, required=True, help="Max norm at each step.")
parser.add_argument('--n-ensemble', type=int, default=1, help="Number of samples to ensemble (outer loop). Default: 1")
parser.add_argument('--shuffle', action='store_true', help="Random order of models vs sequential order of the MCMC (default)")
parser.add_argument('--n-random-init', type=int, default=0, help="Number of random restarts to perform. 0: no random init.")
parser.add_argument('--n-random-init-iter', type=int, default=0, help="Number of random restarts to perform at each iteration. 0: no random init.")

parser.add_argument('--n-models-cycle', type=int, help="Number of models samples per cycle (only used for limit-n-samples-per-cycle or limit-n-cycles)")
parser.add_argument('--limit-n-samples-per-cycle', type=int, default=None, help="Takes into account only the first n samples inside a cycle, droping off the last ones. Default: None (desactivated)")
parser.add_argument('--method-samples-per-cycle', choices=['interval', 'first', 'last'], default='interval', help="Method to select samples inside cycle")
parser.add_argument('--limit-n-cycles', type=int, default=None, help="Takes into account only the first n cycles, droping off the last ones. Default: None (desactivated)")

# others
parser.add_argument("--n-examples", type=int, default=None, help="Craft adv ex on a subset of test examples. If None "
                                                                 "(default), perturbate all the test set.")
parser.add_argument("--seed", type=int, default=None, help="Set random seed")
parser.add_argument("--iters-metrics", type=int, default=10, help="Compute metrics each provided number of iterations.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")
parser.add_argument("--force-add", type=float, default=None, help="Add this scalar to the example. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--force-divide", type=float, default=None, help="Divide the example ex by this scalar. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--skip-accuracy-computation", action='store_true', help="Do not compute accuracies. To be used for full test set.")

args = parser.parse_args()
if args.norm == 'inf':
    args.norm = np.inf
else:
    args.norm = int(args.norm)

# check args
if args.limit_n_samples_per_cycle or args.limit_n_cycles:
    if not args.n_models_cycle:
        raise ValueError("If a limit is set in the number of models to consider, you have to precise the number of samples per cycle.")

# set random seed
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# detect models
paths_ensembles = [list_models(x) for x in args.dirs_models]
print(f'Ensembles of models detected: {[len(x) for x in paths_ensembles]}')
if any([len(x) != len(paths_ensembles[0]) for x in paths_ensembles]):
    raise NotImplementedError('All ensembles should have the same number of models.')
if args.n_models_cycle:
    if any([len(x) % args.n_models_cycle != 0 for x in paths_ensembles]):
        raise ValueError('Number of models is not a multiple of the number of models per cycle')

# load test data
if 'CIFAR100' in args.dirs_models[0]:
    data = CIFAR100(batch_size=args.batch_size)
elif 'CIFAR10' in args.dirs_models[0]:
    data = CIFAR10(batch_size=args.batch_size)
else:
    raise NotImplementedError('Dataset not supported')
X, y = data.to_numpy(train=False, N=args.n_examples, seed=args.seed)

if args.force_add:
    X += args.force_add
    data.min_pixel_value += args.force_add
    data.max_pixel_value += args.force_add
if args.force_divide:
    X /= args.force_divide
    data.min_pixel_value /= args.force_divide
    data.max_pixel_value /= args.force_divide

# limit cycles or samples per cycles
if args.limit_n_cycles or args.limit_n_samples_per_cycle:
    paths_ensembles_lim = []
    for i_ens, paths_models in enumerate(paths_ensembles):
        paths_ensembles_lim.append([])
        for i, path_model in enumerate(paths_models):
            # stop if limit is set on the number of cycles to consider
            if args.limit_n_cycles:
                if i >= args.limit_n_cycles * args.n_models_cycle:
                    break
            # only add current model for selected indexes
            if args.limit_n_samples_per_cycle:
                # select index (at regular interval, always including the last)
                max_index = args.n_models_cycle-1
                if args.method_samples_per_cycle == 'interval':
                    indexes_to_keep = [int(x.left) for x in pd.interval_range(start=0, end=max_index, periods=args.limit_n_samples_per_cycle-1)] + [max_index]
                elif args.method_samples_per_cycle == 'last':
                    indexes_to_keep = list(range(max_index - args.limit_n_samples_per_cycle+1, max_index+1))
                elif args.method_samples_per_cycle == 'first':
                    indexes_to_keep = list(range(0, args.limit_n_samples_per_cycle))
                else:
                    raise NotImplementedError('Method not supported.')
                if (i % args.n_models_cycle) not in indexes_to_keep:
                    continue
            paths_ensembles_lim[i_ens].append(path_model)
    paths_ensembles = paths_ensembles_lim

# shuffle models
if args.shuffle:
    for paths_models in paths_ensembles:
        shuffle(paths_models)

if len(args.dirs_models) > 1 and args.n_ensemble > 1:
    raise ValueError('Attacking multiple ensembles doesn\'t support n-ensemble arg.')

# create nested list of models (ensemble > model)
# [ens1: [m1, m2, m3, m4], ens2: [m5, m6, m7, m8]]
ensemble_list = []
for i, path_model in enumerate(paths_ensembles[0]):
    # if we have multiple MCMC chains, we ensemble
    if len(paths_ensembles) > 1:
        ensemble_list.append([x[i] for x in paths_ensembles])
    else:
    # if args.n_ensemble, we ensemble models from the same MCMC chain
        if len(ensemble_list) == 0:
            # avoid IndexError at first iteration
            ensemble_list.append([path_model, ])
        elif len(ensemble_list[-1]) >= args.n_ensemble:
            ensemble_list.append([path_model, ])
        else:
            ensemble_list[-1].append(path_model)

# create stats df
df_metrics = pd.DataFrame(columns=['n_restart', 'iter', 'acc_ensemble_prob', 'acc_ensemble_logit',
                                   'norm_mean', 'norm_min', 'norm_max', 'time'])
# compute benign acc
if args.skip_accuracy_computation:
    acc_ens_prob, acc_ens_logit = None, None
else:
    acc_ens_prob, acc_ens_logit = compute_accuracy_multiple_ensemble(models_dirs=args.dirs_models, X=X, y=y, data=data)
    print(f"Accuracy on ensemble benign test examples: {acc_ens_prob*100:.3f}% (prob ens),  {acc_ens_logit*100:.3f}% (logit ens)")
df_metrics = df_metrics.append(
    {'n_restart': None, 'iter': None, 'acc_ensemble_prob': acc_ens_prob, 'acc_ensemble_logit': acc_ens_logit,
     'norm_mean': None, 'norm_min': None, 'norm_max': None, 'time': 0.},
    ignore_index=True)

# time code
if USE_CUDA:
    torch.cuda.synchronize()
start_time = time.perf_counter()

acc_worst_restart, restart_best, i_best_restart = 1., None, None
for n_restart in range(max(1, args.n_random_init)):
    print(f'Restart #{n_restart}')
    # random init
    X_adv_tmp = X.copy()
    if args.n_random_init >= 1:
        n = X_adv_tmp.shape[0]
        m = np.prod(X_adv_tmp.shape[1:])
        X_adv_tmp = X_adv_tmp + (
            random_sphere(n, m, radius=args.max_norm, norm=args.norm).reshape(X_adv_tmp.shape).astype(ART_NUMPY_DTYPE)
        )
        X_adv_tmp = np.clip(X_adv_tmp, data.min_pixel_value, data.max_pixel_value)
    # PGD
    acc_worst, i_best, X_adv_tmp_best = 1., None, X
    for i, ensemble_path in enumerate(cycle(ensemble_list)):
        if i >= args.n_iter:
            break
        # only 1 model to attack
        if len(ensemble_path) == 1:
            model = guess_and_load_model(ensemble_path[0], data=data)
            classifier = load_classifier(model, data=data)
        # if ensembling, store path_model to a list and build the ensembling model
        else:
            models_to_ensemble = []
            for j, path_model in enumerate(ensemble_path):
                # load next model and continue only if ensemble is done
                models_to_ensemble.append(guess_and_load_model(path_model, data=data, force_cpu=False))
            classifier = load_classifier_ensemble(models_to_ensemble, data=data)
        # create attack
        attack = FastGradientMethod(estimator=classifier, targeted=False, norm=args.norm, eps=args.norm_step,
                                    num_random_init=args.n_random_init_iter, batch_size=args.batch_size)
        X_adv_tmp = attack.generate(x=X_adv_tmp, y=y)
        # clean previous attack call
        if len(ensemble_path) > 1:
            del models_to_ensemble, classifier, attack
            if USE_CUDA:
                cuda.empty_cache()
        # project on ball of max_norm size, and clip
        X_adv_tmp = X + projection(X_adv_tmp - X, eps=args.max_norm, norm_p=args.norm)  # project on the ball
        X_adv_tmp = np.clip(X_adv_tmp, data.min_pixel_value, data.max_pixel_value)

        # print and save stats
        if ((i % args.iters_metrics == 0) or (i + 1 == args.n_iter)) and not args.skip_accuracy_computation:
            acc_ens_prob, acc_ens_logit = compute_accuracy_multiple_ensemble(models_dirs=args.dirs_models, X=X_adv_tmp, y=y, data=data)
            lpnorm = compute_norm(X_adv=X_adv_tmp, X=X, norm=args.norm)
            if USE_CUDA:
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(
                f"Iteration: {i + 1}/{args.n_iter}, Accuracy ens (prob): {acc_ens_prob * 100:.3f}%, Accuracy ens "
                f"(logit): {acc_ens_logit * 100:.3f}%, L{args.norm}-norm: mean {lpnorm.mean():.5f} (min {lpnorm.min():.5f} max {lpnorm.max():.5f})"
                f", Time: {(end_time - start_time) / 60:.3f} min")
            df_metrics = df_metrics.append(
                {'n_restart': n_restart, 'iter': i, 'acc_ensemble_prob': acc_ens_prob, 'acc_ensemble_logit': acc_ens_logit,
                 'norm_mean': lpnorm.mean(), 'norm_min': lpnorm.min(), 'norm_max': lpnorm.max(), 'time': end_time - start_time},
                ignore_index=True)
            # keep the best adv ex, achieving lowest ensemble accuracies (take the max of the 2 accs to have stronger baseline)
            if acc_worst > max(acc_ens_prob, acc_ens_logit):
                X_adv_tmp_best, i_best, acc_worst = X_adv_tmp, i, max(acc_ens_prob, acc_ens_logit)
    if not args.skip_accuracy_computation:
        print(f'[Restart #{n_restart}] best iteration: {1+i_best}, with test acc: {acc_worst*100} %')
    if acc_worst < acc_worst_restart or args.skip_accuracy_computation:
        acc_worst_restart, restart_best, i_best_restart = acc_worst, n_restart, i_best
        X_adv_best = X_adv_tmp_best
        X_adv = X_adv_tmp

if not args.skip_accuracy_computation:
    print(f'Lowest accuracies achieved at restart #{restart_best} during iteration {1+i_best_restart}: {acc_worst_restart * 100} %')

model_name_list = [guess_model(x) for x in args.dirs_models]
if len(args.dirs_models) > 1:
    path_base = f"X_adv/{data.__class__.__name__}/multi{'_'.join(model_name_list)}/{os.path.split(args.dirs_models[0])[1]}"
else:
    path_base = args.dirs_models[0].replace("models", 'X_adv')
relative_path = f'PGDens_ens{args.n_ensemble}_niter{args.n_iter}_nrestart{args.n_random_init}iter{args.n_random_init_iter}_shuffle{args.shuffle}'
filename = f'Xadv_maxL{args.norm}norm{args.max_norm}_normstep{args.norm_step}_imetric{args.iters_metrics}_n{args.n_examples}_seed{args.seed}'
if args.limit_n_cycles or args.limit_n_samples_per_cycle:
    filename += f'_limcy{args.limit_n_cycles}spc{args.limit_n_samples_per_cycle}'
    if args.method_samples_per_cycle != "interval":
        filename += f"method{args.method_samples_per_cycle}"
path = os.path.join(path_base, relative_path)
save_numpy(array=X_adv, path=path, filename=filename + '.npy')
save_numpy(array=X_adv_best, path=path, filename=filename + '_best.npy')
save_numpy(array=X, path=os.path.join(path, 'save'), filename=filename.replace('Xadv_', 'X_')+'.npy')
save_numpy(array=y, path=os.path.join(path, 'save'), filename=filename.replace('Xadv_', 'y_')+'.npy')
df_metrics.to_csv(os.path.join(path, filename + '_metrics.csv'))
