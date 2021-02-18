import torch
import argparse
import os
import math
import numpy as np
import pandas as pd
import time
from utils.optimizers import SGLD, pSGLD
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from utils.data import MNIST, CIFAR10, CIFAR100, ImageNet
from utils.models import MnistFc, MnistCnn, CifarLeNet, TorchEnsemble
from pytorch_ensembles import models as pemodels
from torchvision.models import resnet50
from utils.helpers import args2paths, load_list_models, MCMC_OPTIMIZERS, ALL_MODELS_NAMES, PEMODELS_NAMES, USE_CUDA

cudnn.benchmark = True
cudnn.deterministic = True


def main(args):
    if args.pretrained and args.dataset != 'ImageNet':
        raise NotImplementedError('Pretrained only implemented for ImageNet')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu' and args.verbose >= 1:
        print("[WARNING] No GPU available")
    # fix random seed
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    SAMPLING = args.optimizer in MCMC_OPTIMIZERS  # True: MCMC, False: DNN
    # create stats df
    df_metrics = pd.DataFrame(columns=['model_id', 'epoch', 'loss', 'acc_test', 'time'])
    # load data
    if args.dataset == 'MNIST':
        data = MNIST(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'CIFAR10':
        data = CIFAR10(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'CIFAR100':
        data = CIFAR100(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'ImageNet':
        data = ImageNet(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise NotImplementedError('Not supported dataset')
    # compute variables
    if SAMPLING:
        # convert burnin and sampling interval in number of epochs to numbers of mini-batch if specified
        if args.burnin_epochs:
            args.burnin = args.burnin_epochs * len(data.trainloader)
        if args.sampling_interval_epochs:
            args.sampling_interval = args.sampling_interval_epochs * len(data.trainloader)
        # compute number of epochs needed
        num_epochs = math.ceil((args.burnin + args.samples * args.sampling_interval) / len(data.trainloader))
        num_restarts = 1
        if args.verbose >= 2:
            print(f'Training for {num_epochs} epochs...')
    else:
        num_epochs = args.epochs
        num_restarts = args.models
    for i_model in range(num_restarts):
        if num_restarts > 1 and args.verbose >= 1:
            print(f'Training model #{i_model}...')
        # load model
        if args.dataset == 'MNIST' and args.architecture == 'CNN':
            model_class = MnistCnn
        elif args.dataset == 'MNIST' and args.architecture == 'FC':
            model_class = MnistFc
        elif args.dataset in ['CIFAR10', 'CIFAR100'] and args.architecture == 'LeNet':
            model_class = CifarLeNet
        elif args.dataset in ['ImageNet'] and args.architecture == 'resnet50':
            model_class = resnet50
        elif args.dataset in ['CIFAR10'] and args.architecture in PEMODELS_NAMES:
            model_class = getattr(pemodels, args.architecture)
        else:
            raise NotImplementedError('Unsupported architecture for this dataset.')
        if args.architecture in PEMODELS_NAMES:
            model = model_class.base(num_classes=data.num_classes, **model_class.kwargs)
        else:
            model = model_class(num_classes=data.num_classes, pretrained=args.pretrained)
        model.train()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        if USE_CUDA:
            criterion.cuda()
        # load optimizer
        weight_decay = 1 / (args.prior_sigma ** 2)
        if args.optimizer == 'SGD':
            optimizer = SGD(params=model.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
        elif args.optimizer == 'RMSprop':
            optimizer = RMSprop(params=model.parameters(), lr=args.lr, weight_decay=weight_decay, alpha=0.99)
        elif args.optimizer == 'Adam':
            optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=weight_decay)
        elif args.optimizer == 'SGLD':
            optimizer = SGLD(params=model.parameters(), lr=args.lr, prior_sigma=args.prior_sigma)
        elif args.optimizer == 'pSGLD':
            optimizer = pSGLD(params=model.parameters(), lr=args.lr, prior_sigma=args.prior_sigma, alpha=0.99)
        else:
            raise NotImplementedError('Not supported optimizer')
        # set lr decay
        if args.lr_decay_on_plateau:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_gamma, verbose=True)
        elif args.lr_decay:
            scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=args.lr_decay_gamma)
        # print stats if pretrained
        if args.pretrained:
            acc_test = data.compute_accuracy(model=model, train=False)
            if args.verbose >= 2:
                print(
                    f"Epoch 0/{num_epochs}, Loss: None, Accuracy: {acc_test * 100:.3f}, "
                    f"Time: 0 min")
            df_metrics = df_metrics.append(
                {'model_id': i_model, 'epoch': -1, 'loss': None, 'acc_test': acc_test,
                 'time': 0},
                ignore_index=True)
        i = 0  # index current mini-batch
        i_sample = 0  # index sample
        # time code
        if USE_CUDA:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for j, (inputs, labels) in enumerate(data.trainloader):
                model.train()
                inputs, labels = inputs.to(device), labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if list(model.parameters())[0].grad is None:
                    print(f'! mini-batch {j} none grad')
                if (list(model.parameters())[0].grad == 0).all():
                    print(f'! mini-batch {j} grad are all 0')

                # verbose
                if args.verbose >= 3 and (j % 1000) == 0:
                    print(f'[ {j} mini-batch / {len(data.trainloader)} ] partial loss: {loss.item():.3f}')

                # save sample
                model.eval()
                if SAMPLING:
                    if i >= args.burnin and (i - args.burnin) % args.sampling_interval == 0:
                        path_model, path_metrics = args2paths(args=args, index_model=i_sample)
                        torch.save(model.state_dict(), path_model)
                        i_sample += 1
                    if i_sample >= args.samples:
                        break  # stop current epochs if all samples are collected
                i += 1

            # print statistics
            running_loss /= len(data.trainloader)
            acc_train = data.compute_accuracy(model=model, train=True)
            acc_test = data.compute_accuracy(model=model, train=False)
            # time code
            if USE_CUDA:
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            if args.verbose >= 2 or (epoch + 1 == num_epochs):
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.3f}, Accuracy train: {acc_train * 100:.3f}, Accuracy test: {acc_test * 100:.3f}, "
                    f"Time: {(end_time - start_time) / 60:.3f} min")
            df_metrics = df_metrics.append(
                {'model_id': i_model, 'epoch': epoch, 'loss': running_loss, 'acc_test': acc_test,
                 'time': end_time - start_time},
                ignore_index=True)

            # decay LR if set
            if args.lr_decay_on_plateau:
                scheduler.step(acc_test)
            elif args.lr_decay:
                scheduler.step()
        # save final model and df_metrics
        if not SAMPLING:
            if num_restarts == 1:
                path_model, path_metrics = args2paths(args=args, index_model=None)
            else:
                path_model, path_metrics = args2paths(args=args, index_model=i_model)
            torch.save(model.state_dict(), path_model)
    df_metrics.to_csv(path_metrics)
    # compute ensemble accuracy
    if SAMPLING or (num_restarts > 1):
        models_dir = os.path.split(path_model)[0]
        list_models = load_list_models(models_dir=models_dir, class_model=model_class, device=device)
        model_ens = TorchEnsemble(models=list_models, ensemble_logits=False)
        model_ens.to(device)
        acc_ens = data.compute_accuracy(model=model_ens, train=False)
        del model_ens
        if USE_CUDA:
            torch.cuda.empty_cache()
        model_ens2 = TorchEnsemble(models=list_models, ensemble_logits=True)
        model_ens2.to(device)
        acc_ens2 = data.compute_accuracy(model=model_ens2, train=False)
        if args.verbose >= 1:
            print(f'Accuracy ensemble probs : {acc_ens * 100:.3f} \n'
                  f'Accuracy ensemble logits: {acc_ens2 * 100:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an ensemble of neural network.")
    parser.add_argument("output", help="Base directory to save the model(s).")
    parser.add_argument("dataset", choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'], help="Dataset to train.")
    parser.add_argument("architecture", choices=ALL_MODELS_NAMES,
                        help="Type of architecture to use.")
    parser.add_argument("--pretrained", action='store_true', help="Warm start. Use pretrained NN.")

    parser_optim = parser.add_argument_group('Optimizer')
    parser_optim.add_argument("optimizer", choices=['SGD', 'RMSprop', 'Adam', 'SGLD', 'pSGLD'], help="Optimizer to use.")
    parser_optim.add_argument("--batch-size", type=int, default=128, help="Size of mini-batches.")
    parser_optim.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser_optim.add_argument('--lr-decay', type=int, default=None, help='Multiply the LR by lr-decay-gamma each provided number of '
                                                                         'epochs. Default: None, ie. fixed LR.')
    parser_optim.add_argument('--lr-decay-gamma', type=float, default=0.5, help='Coefficient to multiply LR at each update. Default: '
                                                                                '0.5. Only used if. lr-decay or lr-decay-on-plateau is set.')
    parser_optim.add_argument('--lr-decay-on-plateau', action='store_true', help='Use LR decay when the test accuracy plateau for 10 epochs. Default: None, ie. fixed LR.')
    parser_optim.add_argument('--prior-sigma', type=float, default=1.,
                              help='Standard deviation of parameter prior. weight_decay=(1/prior-sigma^2). Default: 1')

    parser_dnn = parser.add_argument_group('DNN', description='Parameters for DNN optimizers (RMSprop, Adam).')
    parser_dnn.add_argument("--models", type=int, default=1, help="Number of models to train indendently to ensemble. "
                                                                  "If 1, only train one model.")
    parser_dnn.add_argument("--epochs", type=int, default=200, help="Number of epochs to perform")

    parser_mcmc = parser.add_argument_group('MCMC', description='Parameters for MCMC optimizers (SGLD, pSGLD).')
    parser_mcmc.add_argument("--samples", type=int, default=100, help="Number of samples from parameters' posterior "
                                                                      "to collect. Should be of the scale of "
                                                                      "N/batch-size.")
    parser_mcmc.add_argument("--sampling-interval", type=int, default=100,
                             help="Collect one sample each provided number of mini-batches.")
    parser_mcmc.add_argument("--sampling-interval-epochs", type=int, default=None,
                             help="Collect one sample each provided number of epochs.")
    parser_mcmc.add_argument("--burnin", type=int, default=300,
                             help="Number of mini-batches to skip before collecting samples.")
    parser_mcmc.add_argument("--burnin-epochs", type=int, default=None,
                             help="Number of epochs to skip before collecting samples.")

    parser.add_argument('--seed', type=int, default=None, help='Fix random seed.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers.')
    parser.add_argument('--verbose', type=int, default=2, help='Level of verbosity.')
    args = parser.parse_args()
    main(args)
