import os
import glob
import csv
import numpy as np
import pandas as pd
import argparse
import torch
from utils.data import CIFAR10, CIFAR100
from utils.helpers import guess_and_load_model, project_on_sphere, compute_norm, load_classifier, DEVICE

parser = argparse.ArgumentParser(description="Test an already crafted adv ex against a provided model")
parser.add_argument("path_model", help="Path to pytorch file of model")
parser.add_argument("--path_x_adv", help="Path to adversarial example as numpy array")
parser.add_argument("--directory_x_adv", help="Directory containing adversarial example as numpy array. Evaluate all numpy arrays recursively in the directory.")
parser.add_argument("--export_csv", help="Export evaluation data in CSV.")
parser.add_argument("--project", action="store_true",  help="Project x_adv on the L2 sphere of size sphere_size. Default: False")
parser.add_argument("--sphere_size", type=float, default=4., help="Value of L2-norm sphere size")
parser.add_argument("--norm", type=int, default=None, help="Lp-norm to compute and use for sphere-size. Default: guess from filename")
parser.add_argument("--force-add", type=float, default=None, help="Add this scalar to the adv ex. Use for compatibility with model trained on other range of pixels")
parser.add_argument("--force-divide", type=float, default=None, help="Divide the adv ex by this scalar. Use for compatibility with model trained on other range of pixels")
args = parser.parse_args()

# load test data
if 'CIFAR100' in args.path_model:
    data = CIFAR100(batch_size=128)
elif 'CIFAR10' in args.path_model:
    data = CIFAR10(batch_size=128)
else:
    raise NotImplementedError('Dataset not supported')
# load target model
model = guess_and_load_model(args.path_model, data=data)
classifier = load_classifier(model=model, data=data)

if args.path_x_adv:
    # load single X_adv to test
    path_x_adv_list = [args.path_x_adv]
elif args.directory_x_adv:
    # load recursively all numpy files in dir
    path_x_adv_list = glob.glob(os.path.join(args.directory_x_adv, '**/Xadv_*.npy'), recursive=True)
    path_x_adv_list.sort()
else:
    raise ValueError('--path_x_adv or --directory_x_adv should be provided')

df_metrics = pd.DataFrame(columns=[
    'model_target',
    'adv_ex_path',
    'acc_benin_ex',
    'acc_all_adv_ex',
    'norm_mean_all',
    'norm_min_all',
    'norm_max_all',
    'attack_fail_rate',
    'norm_mean_predok',
    'norm_min_predok',
    'norm_max_predok',
    'nb_pred_ok',
    'norm_type',
])

for path_x_adv in path_x_adv_list:
    X_adv = np.load(path_x_adv)
    # load regular X
    path_base, filename_xadv = os.path.split(path_x_adv)
    filename_xadv = filename_xadv.replace('_best', '')  # remove _best if in filename
    path_x = os.path.join(path_base, 'save', filename_xadv.replace('Xadv_', 'X_'))
    X = np.load(path_x)
    path_y = os.path.join(path_base, 'save', filename_xadv.replace('Xadv_', 'y_'))
    y = np.load(path_y)
    norm = args.norm
    if not norm:
        if 'L2norm' in filename_xadv:
            norm = 2
        elif 'Linfnorm' in filename_xadv:
            norm = np.inf
        else:
            raise NotImplementedError('Unable to detect norm to use')

    if args.force_add:
        X += args.force_add
        X_adv += args.force_add
        data.min_pixel_value += args.force_add
        data.max_pixel_value += args.force_add
    if args.force_divide:
        X /= args.force_divide
        X_adv /= args.force_divide
        data.min_pixel_value /= args.force_divide
        data.max_pixel_value /= args.force_divide

    # project if set
    if args.project:
        X_adv_original = X_adv.copy()
        X_adv = project_on_sphere(X=X, X_adv=X_adv, data=data, size=args.sphere_size, norm=norm)

    # predict
    y_pred = classifier.predict(X)
    label_pred = np.argmax(y_pred, axis=1)
    y_pred_adv = classifier.predict(X_adv)
    label_pred_adv = np.argmax(y_pred_adv, axis=1)

    acc_benin_ex = np.mean(y == label_pred)
    print(f"\n*** Evaluation of grey-box attacks --- {path_x_adv} \n    against target {args.path_model} ***")
    print(f"Accuracy of benin examples: {acc_benin_ex * 100:.3f} %")
    if args.project:
        lpnorm_original = np.linalg.norm(X.reshape((X.shape[0], -1)) - X_adv_original.reshape((X_adv_original.shape[0], -1)), ord=norm, axis=1)
        print(f"*Projection* X_adv projected onto sphere fo size {args.sphere_size}. Original mean L{norm} norm: {lpnorm_original.mean():.3f}")
    acc_adv_all = np.mean(y == label_pred_adv)
    #print(f'Accuracy of all adv ex on the target: {acc_adv_all * 100:.3f} %')
    lpnorm_adv_all = compute_norm(X_adv=X_adv, X=X, norm=norm)
    #print(f"L{norm} norm: Mean {lpnorm_adv_all.mean():.3f} ; Range [{lpnorm_adv_all.min():.3f}, {lpnorm_adv_all.max():.3f}]")
    #print("--")

    # only on benin examples correctly classified
    pred_ok = (y == label_pred)
    acc_predok = np.mean(y[pred_ok] == label_pred_adv[pred_ok])
    print(f'Attack fail rate: accuracy of adv ex *only* taking into account benin examples classified correctly by the target: {acc_predok * 100:.3f} %')
    #print(f'Number of benin examples classified correctly by the target: {pred_ok.sum()}')
    lpnorm_predok = compute_norm(X_adv=X_adv[pred_ok,:,:], X=X[pred_ok,:,:], norm=norm)
    print(f"L{norm} norm: Mean {lpnorm_predok.mean():.3f} ; Range [{lpnorm_predok.min():.3f}, {lpnorm_predok.max():.3f}]")
    print("--")
    df_metrics = df_metrics.append(
        {'model_target': args.path_model,
         'adv_ex_path': path_x_adv,
         'acc_benin_ex': acc_benin_ex,
         'acc_all_adv_ex': acc_adv_all,
         'norm_mean_all': lpnorm_adv_all.mean(),
         'norm_min_all': lpnorm_adv_all.min(),
         'norm_max_all': lpnorm_adv_all.max(),
         'attack_fail_rate': acc_predok,
         'norm_mean_predok': lpnorm_predok.mean(),
         'norm_min_predok': lpnorm_predok.min(),
         'norm_max_predok': lpnorm_predok.max(),
         'nb_pred_ok': pred_ok.sum(),
         'norm_type': norm},
        ignore_index=True)

if args.export_csv:
    df_metrics.to_csv(args.export_csv, quoting=csv.QUOTE_NONNUMERIC)