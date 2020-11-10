"""
First attack implementation to experiment different ways to iterate on samples (w/wo cycle structure,
w/wo shuffle, w/wo random init, several iteration per model vs. only 1, etc.)
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
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.utils import projection, random_sphere
from art.config import ART_NUMPY_DTYPE
from utils.data import CIFAR10, CIFAR100
from utils.helpers import guess_and_load_model, load_classifier, load_classifier_ensemble, list_models, compute_accuracy_ensemble, save_numpy, compute_norm, USE_CUDA
import torch.backends.cudnn as cudnn


cudnn.benchmark = True
cudnn.deterministic = True

# parse args
parser = argparse.ArgumentParser(description="Craft attack against an ensemble of models trained with cSGLD")
parser.add_argument("dir_models",  help="Path to directory containing all the models file of the ensemble model")
parser.add_argument('n_models_cycle', type=int, default=3, help="Number of models samples per cycle")
# attack parameter
parser.add_argument("attack_name", choices=['FGM', 'PGD'], help="Attack to apply to each model name can be either: FGM, PGD")
parser.add_argument('--n-iter-attack', type=int, default=1, help="Number of iteration passed to the attack (supported only for PGD)")
parser.add_argument("--norm", choices=['1', '2', 'inf'], default='2', help="Type of L-norm to use. Default: 2")
parser.add_argument("--max-norm", type=float, required=True, help="Max L2 norm of the perturbation")
parser.add_argument("--norm-inner", type=float, required=True, help="Norm of the perturbation passed to one attack call (inner-loop)")
#parser.add_argument("--norm-outer", type=float, help="Norm of the perturbation passed to one attack call (outer-loop)")  # TODO ?
# parameters to setup different attacks
parser.add_argument('--n-iter-inner', type=int, default=1, help="Number of times to apply the attack inside a cycle (inner loop). Default: 1")
parser.add_argument('--n-iter-outer', type=int, default=10, help="Number of cycles to attack (outer loop). Default: 10")

parser.add_argument('--ensemble-inner', type=int, default=1, help="Number of models to ensemble inside a cycle (inner loop). Default: 1")
parser.add_argument('--ensemble-outer', type=int, default=1, help="Number of cycles to ensemble (outer loop). Default: 1")

parser.add_argument('--shuffle-inner', action='store_true', help="Random order of models inside a cycle (inner loop) vs sequential order of the MCMC (default)")
parser.add_argument('--shuffle-outer', action='store_true', help="Random order of cycles (outer loop) vs sequential order of the MCMC (default)")

parser.add_argument('--n-random-init-inner', type=int, default=0, help="Number of random restarts to perform when applying the attack (inside the inner loop).")
parser.add_argument('--n-random-init-outer', type=int, default=0, help="Number of random restarts to perform (before the outer loop). If 0, no random initialization is performed (default).")

parser.add_argument('--limit-first-n-inner', type=int, default=None, help="Takes into account only the first n samples inside a cycle (inner loop), droping off the last ones. Default: None (desactivated)")
parser.add_argument('--limit-first-n-outer', type=int, default=None, help="Takes into account only the first n cycles (outer loop), droping off the last ones. Default: None (desactivated)")

# others
parser.add_argument("--n-examples", type=int, default=None, help="Craft adv ex on a subset of test examples. If None "
                                                                 "(default), perturbate all the test set.")
parser.add_argument("--seed", type=int, default=None, help="Set random seed")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")
parser.add_argument("--force-add", type=float, default=None, help="Add this scalar to the example. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--force-divide", type=float, default=None, help="Divide the example ex by this scalar. Use for compatibility with model trained on other range of pixels")

args = parser.parse_args()
if args.norm == 'inf':
    args.norm = np.inf
else:
    args.norm = int(args.norm)

# set random seed
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# check
if args.n_iter_attack > 1 and args.attack_name in ['FGM',]:
    raise ValueError('single single step attack doesn\'t support n-iter-attack arg')

# detect models
path_models = list_models(args.dir_models)
print('Ensemble of {} models detected'.format(len(path_models)))
if len(path_models) % args.n_models_cycle != 0:
    raise ValueError('Number of models is not a multiple of the number of models per cycle')

# load test data
if 'CIFAR100' in args.dir_models:
    data = CIFAR100(batch_size=args.batch_size)
elif 'CIFAR10' in args.dir_models:
    data = CIFAR10(batch_size=args.batch_size)
else:
    raise NotImplementedError('Dataset not supported')
X, y = data.to_numpy(train=False, N=args.n_examples)

if args.force_add:
    X += args.force_add
    data.min_pixel_value += args.force_add
    data.max_pixel_value += args.force_add
if args.force_divide:
    X /= args.force_divide
    data.min_pixel_value /= args.force_divide
    data.max_pixel_value /= args.force_divide

# create nested list of models (cycles > models)
# [cycle1: [m1, m2, m3, m4], cycle2: [m5, m6, m7, m8]]
path_cycles = []
for i, path_model in enumerate(path_models):
    if i % args.n_models_cycle == 0:
        path_cycles.append([path_model, ])
    else:
        path_cycles[i // args.n_models_cycle].append(path_model)

# limit cycles to first ones if provided
if args.limit_first_n_outer:
    if args.limit_first_n_outer > len(path_cycles):
        raise ValueError('limit_first_n_outer is greater than number of cycles')
    print(f"Only takes into account the first {args.limit_first_n_outer} cycles.")
    path_cycles = path_cycles[0:args.limit_first_n_outer]
# limit samples per each cycle to first ones if provided
if args.limit_first_n_inner:
    if args.limit_first_n_inner > len(path_cycles[0]):
        raise ValueError('limit_first_n_inner is greater than number of samples per cycle')
    print(f"Only takes into account the first {args.limit_first_n_inner} samples per cycle.")
    for i, cycle_list in enumerate(path_cycles):
        path_cycles[i] = cycle_list[0:args.limit_first_n_inner]

# shuffle cycles
if args.shuffle_outer:
    shuffle(path_cycles)

# random init
X_adv_tmp = X.copy()
if args.n_random_init_outer == 1:
    n = X_adv_tmp.shape[0]
    m = np.prod(X_adv_tmp.shape[1:])
    X_adv_tmp = X_adv_tmp + (
        random_sphere(n, m, radius=args.max_norm, norm=args.norm).reshape(X_adv_tmp.shape).astype(ART_NUMPY_DTYPE)
    )
    X_adv_tmp = np.clip(X_adv_tmp, data.min_pixel_value, data.max_pixel_value)
elif args.n_random_init_outer > 1:
    raise NotImplementedError("Multiple random restarts not implemented yet. Try 0 or 1.")

if args.ensemble_outer > 1:
    raise NotImplementedError('Ensembling cycles not implemented yet.')

# create stats df
df_metrics = pd.DataFrame(columns=['outer_iter', 'inner_iter', 'acc_ensemble_prob', 'acc_ensemble_logit',
                                   'norm_mean', 'norm_min', 'norm_max', 'time'])
# compute benign acc
acc_ens_prob, acc_ens_logit = compute_accuracy_ensemble(models_dir=args.dir_models, X=X, y=y, data=data)
print(f"Accuracy on ensemble benign test examples: {acc_ens_prob*100:.3f}% (prob ens),  {acc_ens_logit*100:.3f}% (logit ens)")
lpnorm = compute_norm(X_adv=X_adv_tmp, X=X, norm=args.norm)
df_metrics = df_metrics.append(
    {'outer_iter': 0, 'inner_iter': 0, 'acc_ensemble_prob': acc_ens_prob, 'acc_ensemble_logit': acc_ens_logit,
     'norm_mean': lpnorm.mean(), 'norm_min': lpnorm.min(), 'norm_max': lpnorm.max(), 'time': 0.},
    ignore_index=True)

# time code
if USE_CUDA:
    torch.cuda.synchronize()
start_time = time.perf_counter()

acc_worst, i_best = 1., 0

for i, cycle_list in enumerate(cycle(path_cycles)):
    if i >= args.n_iter_outer:
        break
    print(f'attacking cycle #{i % len(path_cycles)}')
    # shuffle samples at each cycles (useful if the same cycle is called multiple times)
    if args.shuffle_inner:
        shuffle(cycle_list)
    models_to_ensemble = []
    for j, path_model in enumerate(cycle(cycle_list)):
        if j >= args.n_iter_inner * args.ensemble_inner:
            break
        # if no ensembling, simply load the model to the classifier
        if args.ensemble_inner == 1:
            model = guess_and_load_model(path_model, data=data)
            classifier = load_classifier(model, data=data)
        # if ensembling, store path_model to a list and build the ensembling model
        elif args.ensemble_inner >= 2:
            # clean previous attack call
            if len(models_to_ensemble) >= args.ensemble_inner:
                del models_to_ensemble, classifier, attack
                if USE_CUDA:
                    cuda.empty_cache()
                models_to_ensemble = []
            # load next model and continue only if ensemble is done
            models_to_ensemble.append(guess_and_load_model(path_model, data=data, force_cpu=False))
            if len(models_to_ensemble) < args.ensemble_inner:
                continue
            classifier = load_classifier_ensemble(models_to_ensemble, data=data)
        else:
            raise ValueError('incorrect ensemble_inner arg')
        # create attack
        if args.attack_name == 'FGM':
            attack = FastGradientMethod(estimator=classifier, targeted=False, norm=args.norm, eps=args.norm_inner,
                                        num_random_init=args.n_random_init_inner, batch_size=args.batch_size)
        elif args.attack_name == 'PGD':
            attack = ProjectedGradientDescent(estimator=classifier, targeted=False, max_iter=args.n_iter_attack, norm=args.norm,
                                              eps=args.norm_inner,
                                              eps_step=args.norm_inner / 4,  # TODO: tune?
                                              num_random_init=args.n_random_init_inner, batch_size=args.batch_size)
        else:
            raise NotImplementedError('attack-name not supported')
        X_adv_tmp = attack.generate(x=X_adv_tmp, y=y)
        # project on ball of max_norm size, and clip
        X_adv_tmp = X + projection(X_adv_tmp - X, eps=args.max_norm, norm_p=args.norm)  # project on the ball
        X_adv_tmp = np.clip(X_adv_tmp, data.min_pixel_value, data.max_pixel_value)

    # print and save stats
    acc_ens_prob, acc_ens_logit = compute_accuracy_ensemble(models_dir=args.dir_models, X=X_adv_tmp, y=y, data=data)
    lpnorm = compute_norm(X_adv=X_adv_tmp, X=X, norm=args.norm)
    if USE_CUDA:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(
        f"Outer iteration: {i + 1}/{args.n_iter_outer}, Accuracy ens (prob): {acc_ens_prob * 100:.3f}%, Accuracy ens "
        f"(logit): {acc_ens_logit * 100:.3f}%, L{args.norm}-norm: mean {lpnorm.mean():.5f} (min {lpnorm.min():.5f} max {lpnorm.max():.5f})"
        f", Time: {(end_time - start_time) / 60:.3f} min")
    df_metrics = df_metrics.append(
        {'outer_iter': i, 'inner_iter': j, 'acc_ensemble_prob': acc_ens_prob, 'acc_ensemble_logit': acc_ens_logit,
         'norm_mean': lpnorm.mean(), 'norm_min': lpnorm.min(), 'norm_max': lpnorm.max(), 'time': end_time - start_time},
        ignore_index=True)
    # keep the best adv ex, achieving lowest ensemble accuracies (take the max of the 2 accs to have stronger baseline)
    if acc_worst > max(acc_ens_prob, acc_ens_logit):
        X_adv_best, i_best, acc_worst = X_adv_tmp, i, max(acc_ens_prob, acc_ens_logit)
    # todo: best at global level, but best can be computed on each examples too with y_pred

X_adv = X_adv_tmp
print(f'Lowest accuracies achieved at outer iteration #{i_best}: {acc_worst * 100} %')

path_base = args.dir_models.replace("models", 'X_adv')
relative_path = f'{args.attack_name}_ensout{args.ensemble_outer}in{args.ensemble_inner}_niterout{args.n_iter_outer}in{args.n_iter_inner}att{args.n_iter_attack}_nraninitout{args.n_random_init_outer}in{args.n_random_init_inner}_shuffleout{args.shuffle_outer}in{args.shuffle_inner}'
filename = f'Xadv_maxL{args.norm}norm{args.max_norm}_normiter{args.norm_inner}_n{args.n_examples}_seed{args.seed}'
if args.limit_first_n_outer or args.limit_first_n_inner:
    filename += f'_limout{args.limit_first_n_outer}in{args.limit_first_n_inner}'
path = os.path.join(path_base, relative_path)
save_numpy(array=X_adv, path=path, filename=filename + '.npy')
save_numpy(array=X_adv_best, path=path, filename=filename + '_best.npy')
save_numpy(array=X, path=os.path.join(path, 'save'), filename=filename.replace('Xadv_', 'X_')+'.npy')
save_numpy(array=y, path=os.path.join(path, 'save'), filename=filename.replace('Xadv_', 'y_')+'.npy')
df_metrics.to_csv(os.path.join(path, filename + '_metrics.csv'))
