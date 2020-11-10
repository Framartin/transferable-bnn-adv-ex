import os
import glob
import torch
import numpy as np
from art.classifiers import PyTorchClassifier
from .models import TorchEnsemble, CifarLeNet

from pytorch_ensembles import models as pemodels
from torchvision.models import resnet50

MCMC_OPTIMIZERS = ['SGLD', 'pSGLD']
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
PEMODELS_NAMES = ['PreResNet110', 'PreResNet164', 'VGG16BN', 'VGG19BN', 'WideResNet28x10',
                          'WideResNet28x10do']
ALL_MODELS_NAMES = ['FC', 'CNN', 'LeNet', 'resnet50'] + PEMODELS_NAMES


def args2paths(args, index_model=None):
    """
    Create a string corresponding to path to save current model
    :param args: train.py command line arguments namespace
    :param index_model: int for a specific element of ensemble models or for a model sample
    :return: set of pytorch model path and metrics CSV file path
    """
    if args.optimizer in MCMC_OPTIMIZERS:
        filename = f'{index_model:04}.pth'
        filename_metrics = 'metrics.csv'
        relative_path = f'{args.dataset}/{args.architecture}/mcmc_samples/{args.optimizer}_bs{args.batch_size}_lr{args.lr}_lrd{"plateau" if args.lr_decay_on_plateau else args.lr_decay}_psig{args.prior_sigma}_s{args.samples}_si{args.sampling_interval}_bi{args.burnin}_seed{args.seed}'
    else:  # DNN
        if index_model is None:  # single DNN
            filename = f'model_{args.optimizer}_bs{args.batch_size}_lr{args.lr}_lrd{"plateau" if args.lr_decay_on_plateau else args.lr_decay}_psig{args.prior_sigma}_ep{args.epochs}_seed{args.seed}.pth'
            filename_metrics = filename.replace('model', 'metrics').replace('.pth', '.csv')
            relative_path = f'{args.dataset}/{args.architecture}/single_model/'
        else:  # Ensemble of DNN
            filename = f'{index_model:04}.pth'
            filename_metrics = 'metrics.csv'
            relative_path = f'{args.dataset}/{args.architecture}/dnn_ensemble/{args.optimizer}_bs{args.batch_size}_lr{args.lr}_lrd{"plateau" if args.lr_decay_on_plateau else args.lr_decay}_psig{args.prior_sigma}_ep{args.epochs}_seed{args.seed}'
    os.makedirs(os.path.join(args.output, relative_path), exist_ok=True)
    return os.path.join(args.output, relative_path, filename), \
           os.path.join(args.output, relative_path, filename_metrics)


def list_models(models_dir):
    path_models = glob.glob(f'{models_dir}/*.pt')
    path_models.extend(glob.glob(f'{models_dir}/*.pth'))
    path_models = sorted(path_models)
    return path_models


def load_model(path_model, class_model, *args, **kwargs):
    model = class_model(*args, **kwargs)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    return model


def load_list_models(models_dir, class_model, device=None, *args, **kwargs):
    path_models = list_models(models_dir)
    models = []
    for path_model in path_models:
        model = load_model(path_model=path_model, class_model=class_model)
        if device:
            model.to(device)
        models.append(model)
    return models


def guess_model(path_model):
    """
    Return the name of the model
    """
    candidates = [x for x in ALL_MODELS_NAMES if x in path_model]
    if len(candidates) != 1:
        raise ValueError('Not able to guess model name')
    return candidates[0]


def guess_and_load_model(path_model, data, force_cpu=False):
    """
    Load model from its path only (guessing the model class)
    :param path_model: str, path to the pt file to load
    :param data: data class
    :param force_cpu: don't send model to GPU if True
    :return: pytorch instance of a model
    """
    # model from torchvision
    if 'resnet50' in path_model:
        model = resnet50(pretrained=False, num_classes=data.num_classes)
    # model from utils/models.py
    elif 'LeNet' in path_model:
        model = CifarLeNet(pretrained=False, num_classes=data.num_classes)
    # model from pytorch-ensembles
    elif len([x for x in PEMODELS_NAMES if x in path_model]) >= 1:
        # list model name in pemodels: [x for x in dir(pemodels) if x[0:2] != '__']
        model_name_list = [x for x in PEMODELS_NAMES if x in path_model]
        if len(model_name_list) != 1:
            raise ValueError(f'Failed to extract model name: {model_name_list}')
        arch = getattr(pemodels, model_name_list[0])
        model = arch.base(num_classes=data.num_classes, **arch.kwargs)
    else:
        raise NotImplementedError('Model class unknown')
    model.load_state_dict(torch.load(path_model, map_location=DEVICE))
    model.eval()
    # to GPU
    if USE_CUDA and not force_cpu:
        model.to(DEVICE)
    return model


def load_classifier(model, data):
    """
    Load ART PyTorch classifier from pytorch model
    :param model: pytorch model instance
    :param data: data class
    :return: ART classifier
    """
    # not used but mandatory for ART
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(data.min_pixel_value, data.max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=tuple(data.trainset.data.shape[1:]),
        nb_classes=data.num_classes,
        device_type="gpu" if USE_CUDA else "cpu"
    )
    classifier.set_learning_phase(False)
    return classifier


def load_classifier_ensemble(models, **kwargs):
    """
    Build an ART classifier of an ensemble of PyTorch models
    :param models: list of pytorch model instances
    :return:
    """
    model = TorchEnsemble(models=models, ensemble_logits=True)
    if USE_CUDA:
        model.to(DEVICE)
    model.eval()
    return load_classifier(model, **kwargs)


def predict_ensemble(models_dir, X, data):
    """
    Compute prediction for each model inside models_dir

    :param models_dir: str path to pytorch's models
    :param X: pytorch tensor or numpy array
    :param data: data class instance
    :return: tuple of 2 numpy arrays of predicted labels with probs and logits ensembling
    """
    if not torch.is_tensor(X):
        X = torch.Tensor(X)
    if USE_CUDA:
        X = X.to(DEVICE)
    path_models = list_models(models_dir)
    y_pred_ens_logit = torch.zeros((X.shape[0], data.num_classes))
    y_pred_ens_prob = torch.zeros((X.shape[0], data.num_classes))
    for path_model in path_models:
        model = guess_and_load_model(path_model=path_model, data=data)
        with torch.no_grad():
            output = model(X)
        y_pred_ens_logit += output.cpu()
        y_pred_ens_prob += torch.nn.functional.softmax(output, dim=1).cpu()
        # clean
        del model, output
        if USE_CUDA:
            torch.cuda.empty_cache()
    y_pred_ens_logit /= len(path_models)
    y_pred_ens_logit = torch.nn.functional.softmax(y_pred_ens_logit, dim=1)
    y_pred_ens_prob /= len(path_models)
    label_pred_logit = np.argmax(y_pred_ens_logit.numpy(), axis=1)
    label_pred_prob = np.argmax(y_pred_ens_prob.numpy(), axis=1)
    return label_pred_prob, label_pred_logit

def compute_accuracy_ensemble(models_dir, X, y, data):
    label_pred_prob, label_pred_logit = predict_ensemble(models_dir=models_dir, X=X, data=data)
    acc_prob = (label_pred_prob == y).mean()
    acc_logit = (label_pred_logit == y).mean()
    return acc_prob, acc_logit


def compute_accuracy_multiple_ensemble(models_dirs, X, y, data):
    """
    Compute the mean of the accuracies of several ensembles.
    """
    acc_prob, acc_logit = 0., 0.
    for models_dir in models_dirs:
        acc_prob_tmp, acc_logit_tmp = compute_accuracy_ensemble(models_dir=models_dir, X=X, y=y, data=data)
        acc_prob += acc_prob_tmp
        acc_logit += acc_logit_tmp
    return acc_prob / len(models_dirs), acc_logit / len(models_dirs)


def save_numpy(array, path, filename):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, filename), array)


def compute_norm(X_adv, X, norm=2):
    return np.linalg.norm(X_adv.reshape((X_adv.shape[0], -1)) - X.reshape((X.shape[0], -1)), ord=norm, axis=1)

def project_on_sphere(X, X_adv, data, size=4., norm=2):
    """
    Project on sphere (not the ball!) of specified size
    :param X: np array
    :param X_adv: np array
    :param size:
    :param norm: Lp norm to use. Only 2 implemented
    :return:
    """
    if norm != 2:
        raise NotImplementedError('Only L2 norm implemented')
    lpnorm = compute_norm(X_adv, X, norm=2)
    X_adv_proj = X + size / lpnorm.reshape((X.shape[0], 1, 1, 1)) * (X_adv - X)
    X_adv_proj = np.clip(X_adv_proj, data.min_pixel_value, data.max_pixel_value)
    return X_adv_proj
