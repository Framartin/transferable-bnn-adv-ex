"""
Implementation of the attacks used in the article
"""

import numpy as np
import pandas as pd
import torch
import argparse
import time
import os
import random
from random import shuffle
from utils.data import CIFAR10, CIFAR100, ImageNet
from utils.helpers import guess_model, guess_and_load_model, load_classifier, load_classifier_ensemble, list_models, \
    compute_accuracy_from_nested_list_models, save_numpy, compute_norm, USE_CUDA
from utils.attacks import ExtendedProjectedGradientDescentPyTorch
import torch.backends.cudnn as cudnn


cudnn.benchmark = True
cudnn.deterministic = True

# parse args
parser = argparse.ArgumentParser(description="Craft PGD adv ex with each update computed on a different samples from an ensemble of models trained with cSGLD")
parser.add_argument("dirs_models", nargs='+', help="Path to directory containing all the models file of the ensemble model")
parser.add_argument('--n-iter', type=int, default=None, help="Number of iterations to perform. If None (default), set to the number of samples.")
parser.add_argument("--norm", choices=['1', '2', 'inf'], default='2', help="Type of L-norm to use. Default: 2")
parser.add_argument("--max-norm", type=float, required=True, help="Max L-norm of the perturbation")
parser.add_argument("--norm-step", type=float, required=True, help="Max norm at each step.")
parser.add_argument('--n-ensemble', type=int, default=1, help="Number of samples to ensemble (outer loop). Default: 1")
parser.add_argument('--shuffle', action='store_true', help="Random order of models vs sequential order of the MCMC (default)")
parser.add_argument('--n-random-init', type=int, default=0, help="Number of random restarts to perform. 0: no random init.")

parser.add_argument('--n-models-cycle', type=int, help="Number of models samples per cycle (only used for limit-n-samples-per-cycle or limit-n-cycles)")
parser.add_argument('--limit-n-samples-per-cycle', type=int, default=None, help="Takes into account only the first n samples inside a cycle, droping off the last ones. Default: None (desactivated)")
parser.add_argument('--method-samples-per-cycle', choices=['interval', 'first', 'last'], default='interval', help="Method to select samples inside cycle")
parser.add_argument('--limit-n-cycles', type=int, default=None, help="Takes into account only the first n cycles, droping off the last ones. Default: None (desactivated)")

# test time transferability improvements
parser.add_argument('--ghost-attack', action='store_true', help="Load each model as a Ghost network (default: no model alteration)")
parser.add_argument('--input-diversity', action='store_true', help="Add input diversity to each model (default: no model alteration)")
parser.add_argument('--translation-invariant', action='store_true', help="Apply translation invariance kernel to gradient (default: regular gradient)")
parser.add_argument("--momentum", type=float, default=None, help="Apply momentum to gradients (default: regular gradient)")

# target model
parser.add_argument("--model-target-path", default=None, help="Path to the target model.")
parser.add_argument("--csv-export", default=None, help="Path to CSV where to export data about target.")
parser.add_argument("--export-target-per-iter", action='store_true', help="Export target acc at each iteration in csv-export file. Else 1 line for final data.")

# others
parser.add_argument("--n-examples", type=int, default=None, help="Craft adv ex on a subset of test examples. If None "
                                                                 "(default), perturbate all the test set. If "
                                                                 "model-target-path is set, extract the subset from "
                                                                 "the examples correctly predicted by it.")
parser.add_argument("--data-path", default=None, help="Path of data. Only supported for ImageNet.")
parser.add_argument("--dir-export", default=None, help="Directory to export Xadv.")
parser.add_argument("--seed", type=int, default=None, help="Set random seed")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Try a lower value if out of memory (especially for high values of --ensemble-inner).")
parser.add_argument("--force-add", type=float, default=None, help="Add this scalar to the example. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--force-divide", type=float, default=None, help="Divide the example ex by this scalar. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--skip-accuracy-computation", action='store_true', help="Do not compute accuracies. To be used for full test set.")
parser.add_argument("--no-save", action='store_true', help="Do not store adv ex as numpy files.")

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
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# detect models
paths_ensembles = [list_models(x) for x in args.dirs_models]
print(f'Ensembles of models detected: {[len(x) for x in paths_ensembles]}')
if any([len(x) == 0 for x in paths_ensembles]):
    raise ValueError('Empty model ensemble')
if args.n_models_cycle:
    if any([len(x) % args.n_models_cycle != 0 for x in paths_ensembles]):
        raise ValueError('Number of models is not a multiple of the number of models per cycle')

# load test data
if 'CIFAR100' in args.dirs_models[0]:
    data = CIFAR100(batch_size=args.batch_size)
elif 'CIFAR10' in args.dirs_models[0]:
    data = CIFAR10(batch_size=args.batch_size)
elif 'ImageNet' in args.dirs_models[0]:
    data = ImageNet(batch_size=args.batch_size, path=args.data_path)
else:
    raise NotImplementedError('Dataset not supported')

model_target = None
if args.model_target_path:
    # load target and select n_examples correctly predicted by it
    # target model is loaded with randomization defense for translation invariance
    model_target = guess_and_load_model(path_model=args.model_target_path, data=data, defense_randomization=args.translation_invariant)
    X, y = data.correctly_predicted_to_numpy(model=model_target, train=False, N=args.n_examples, seed=args.seed)
else:
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

if any([len(x) != len(paths_ensembles[0]) for x in paths_ensembles]):
    raise NotImplementedError('All ensembles should have the same number of models.')
print(f'Ensembles of models used: {[len(x) for x in paths_ensembles]}')


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

# load each models and create ART classifier
ensemble_classifiers = []  # list of ART classifiers. Each one has the logits fused
list_ensemble_models = []  # nested list of torch models
for i, ensemble_path in enumerate(ensemble_list):
    # only 1 model to attack
    if len(ensemble_path) == 1:
        model = guess_and_load_model(ensemble_path[0], data=data, load_as_ghost=args.ghost_attack, input_diversity=args.input_diversity)
        classifier = load_classifier(model, data=data)
        list_ensemble_models.append([model])
    # if ensembling, store path_model to a list and build the ensembling model
    else:
        models_to_ensemble = []
        for j, path_model in enumerate(ensemble_path):
            # load next model and continue only if ensemble is done
            models_to_ensemble.append(guess_and_load_model(path_model, data=data, load_as_ghost=args.ghost_attack, input_diversity=args.input_diversity, force_cpu=False))
        classifier = load_classifier_ensemble(models_to_ensemble, data=data)
        list_ensemble_models.append(models_to_ensemble)
    ensemble_classifiers.append(classifier)
    del classifier

# compute benign acc
if not args.skip_accuracy_computation:
    acc_ens_prob = compute_accuracy_from_nested_list_models(list_ensemble=list_ensemble_models, X=X, y=y, data=data)
    print(f"Accuracy on ensemble benign test examples: {acc_ens_prob*100:.3f}%.")

# time code
if USE_CUDA:
    torch.cuda.synchronize()
start_time = time.perf_counter()

attack = ExtendedProjectedGradientDescentPyTorch(
    estimators=ensemble_classifiers, targeted=False, norm=args.norm, eps=args.max_norm, eps_step=args.norm_step,
    max_iter=args.n_iter, num_random_init=args.n_random_init, batch_size=args.batch_size,
    translation_invariant=args.translation_invariant, momentum=args.momentum,
    model_target=model_target if args.export_target_per_iter else None
)
X_adv = attack.generate(x=X, y=y)

if USE_CUDA:
    torch.cuda.synchronize()
end_time = time.perf_counter()

model_name_list = [guess_model(x) for x in args.dirs_models]

# print stats
if not args.skip_accuracy_computation:
    acc_ens_prob_adv = compute_accuracy_from_nested_list_models(list_ensemble=list_ensemble_models, X=X_adv, y=y, data=data)
    lpnorm = compute_norm(X_adv=X_adv, X=X, norm=args.norm)
    print(
        f"Stats after {args.n_iter} iters: Accuracy ens (prob): {acc_ens_prob_adv * 100:.3f}%, "
        f"L{args.norm}-norm: mean {lpnorm.mean():.5f} (min {lpnorm.min():.5f} max {lpnorm.max():.5f}), "
        f"Time: {(end_time - start_time) / 60:.3f} min")
    if args.csv_export:
        if not args.model_target_path:
            raise ValueError('Target model should be specified to export CSV.')
        acc_target_adv = compute_accuracy_from_nested_list_models([[model_target,],], X=X_adv, y=y, data=data)
        acc_target_original = compute_accuracy_from_nested_list_models([[model_target,],], X=X, y=y, data=data)
        print(f"Attack fail rate: {acc_target_adv * 100:.3f} %")
        df_metrics = pd.DataFrame([{
            'model_target': args.model_target_path,
            'arch_target': guess_model(args.model_target_path),
            'surrogate_type': 'cSGLD' if 'cSGLD' in args.dirs_models[0] else 'dnn',
            'surrogate_archs': '_'.join(model_name_list),
            'surrogate_size_ensembles': len(paths_ensembles[0]),  # nb models per arch
            'norm_type': args.norm,
            'norm_max': args.max_norm,
            'norm_step': args.norm_step,
            'n_iter': args.n_iter,
            'n_ensemble': args.n_ensemble,
            'n_random_init': args.n_random_init,
            'momentum': args.momentum,
            'shuffle': args.shuffle,
            'ghost': args.ghost_attack,
            'input_diversity': args.input_diversity,
            'translation_invariant': args.translation_invariant,
            'adv_fail_rate': acc_target_adv,  # X contains only correctly predicted examples
            'adv_sucess_rate': 1-acc_target_adv,
            'adv_norm_mean': lpnorm.mean(),
            'adv_norm_min': lpnorm.min(),
            'adv_norm_max': lpnorm.max(),
            'limit_samples_cycle': args.limit_n_samples_per_cycle,
            'limit_cycles': args.limit_n_cycles,
            'surrogate_acc_original_ex': acc_ens_prob,
            'surrogate_acc_adv_ex': acc_ens_prob_adv,
            'target_acc_original_ex': acc_target_original,
            'acc_original_ex': acc_ens_prob,
            'nb_adv': X_adv.shape[0],
            'time': end_time - start_time,
        },])
        if args.export_target_per_iter:
            # duplicate the df line to the number of iterations
            df_metrics = pd.concat([df_metrics] * args.n_iter, ignore_index=True)
            df_metrics['n_iter'] = list(range(args.n_iter))
            df_metrics['adv_fail_rate'] = attack.get_target_accuracy_per_iter()
            df_metrics['adv_sucess_rate'] = 1 - df_metrics['adv_fail_rate']
        # create dir and append one line to csv
        os.makedirs(os.path.dirname(args.csv_export), exist_ok=True)
        df_metrics.to_csv(args.csv_export, mode='a', header=not os.path.exists(args.csv_export), index=False)


if not args.no_save:
    if args.dir_export:
        path_base = os.path.join(args.dir_export)
        if len(args.dirs_models) > 1:
            path_base = os.path.join(path_base, f"multi{'_'.join(model_name_list)}/{'cSGLD' if 'cSGLD' in args.dirs_models[0] else 'dnn'}")
    elif len(args.dirs_models) > 1:
        path_base = f"X_adv/{data.__class__.__name__}/multi_archs/multi{'_'.join(model_name_list)}/{os.path.split(args.dirs_models[0])[1]}"
    else:
        path_base = args.dirs_models[0].replace("models", 'X_adv')
    relative_path = f'PGDens_ens{args.n_ensemble}_niter{args.n_iter}_nrestart{args.n_random_init}_shuffle{args.shuffle}'
    filename = f'Xadv_maxL{args.norm}norm{args.max_norm}_normstep{args.norm_step}_n{args.n_examples}_seed{args.seed}'
    if args.momentum:
        filename += f'_momentum{args.momentum}'
    if args.limit_n_cycles or args.limit_n_samples_per_cycle:
        filename += f'_limcy{args.limit_n_cycles}spc{args.limit_n_samples_per_cycle}'
        if args.method_samples_per_cycle != "interval":
            filename += f"method{args.method_samples_per_cycle}"
    if args.ghost_attack:
        relative_path = 'ghost/' + relative_path
    if args.input_diversity:
        relative_path = 'input_diversity/' + relative_path
    if args.translation_invariant:
        relative_path = 'translation_invariant/' + relative_path
    path = os.path.join(path_base, relative_path)
    save_numpy(array=X_adv, path=path, filename=filename + '.npy')
    save_numpy(array=X, path=os.path.join(path, 'save'), filename=filename.replace('Xadv_', 'X_')+'.npy')
    save_numpy(array=y, path=os.path.join(path, 'save'), filename=filename.replace('Xadv_', 'y_')+'.npy')
