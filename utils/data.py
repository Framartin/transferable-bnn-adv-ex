import os
import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
from .helpers import list_models, guess_and_load_model, DEVICE


class DataBase:
    trainloader = None
    testloader = None
    transform = None
    classes = None
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, batch_size=64, num_workers=0):
        if hasattr(self, 'min_pixel_value') + hasattr(self, 'max_pixel_value') < 2:
            self.min_pixel_value = 1e8
            self.max_pixel_value = -1e8
            for images, _ in self.testloader:
                min_pixel = torch.min(images)
                max_pixel = torch.max(images)
                if min_pixel < self.min_pixel_value:
                    self.min_pixel_value = min_pixel
                if max_pixel > self.max_pixel_value:
                    self.max_pixel_value = max_pixel
        self.classes = self.testset.classes
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_input_shape(self):
        return tuple(self.trainset.data.shape)

    def to_numpy(self, train=False, N=None, seed=None):
        """
        Return dataset as numpy array
        Becareful, data has to be able to be loaded into memory.
        :param train: bool, train or test set
        :param N: int, max number of examples to import
        :return: X, y: numpy arrays
        """
        if train:
            set = self.trainset
        else:
            set = self.testset
        if N is None:
            N = len(set)
        if seed:
            torch.manual_seed(seed)
        loader = torch.utils.data.DataLoader(set, batch_size=N, shuffle=(train or N < len(set)))
        load_tmp = next(iter(loader))
        X = load_tmp[0].numpy()
        y = load_tmp[1].numpy()
        return X, y

    def correctly_predicted_to_numpy(self, model, train=False, N=None, seed=None):
        """
        Return the examples correcty predicted by model in the dataset as numpy arrays

        :param model: pytorch model
        :param train: bool, train or test set
        :param N: int, max number of examples to import
        :return: X, y: numpy arrays
        """
        if train:
            set = self.trainset
        else:
            set = self.testset
        if N is None:
            N = len(set)
        if seed:
            torch.manual_seed(seed)
        loader = torch.utils.data.DataLoader(set, batch_size=self.batch_size, shuffle=(train or N < len(set)))
        X = np.zeros((N,) + self.get_input_shape()[1:], dtype='float32')
        y = np.zeros((N,), dtype='int')
        nb_images_saved = 0
        for images, labels in loader:
            images_ = images.to(DEVICE)
            outputs = model(images_)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            images_ok = images[predicted == labels,]
            labels_ok = labels[predicted == labels,]
            nb_images_to_append = min((predicted == labels).sum().item(), N-nb_images_saved)
            X[nb_images_saved:(nb_images_saved+nb_images_to_append),] = images_ok[0:nb_images_to_append,].numpy()
            y[nb_images_saved:nb_images_saved+nb_images_to_append] = labels_ok[0:nb_images_to_append].numpy()
            nb_images_saved += nb_images_to_append
            if nb_images_saved >= N:
                break
        if not (X.shape[0] == y.shape[0] == N):
            raise RuntimeError("Array shape unexpected")
        return X, y

    def compute_accuracy(self, model, train=False):
        """
        Compute the accuracy on the test or train data
        :param model: Pytorch NN
        :param train: compute on the test or train set
        :return: float
        """
        loader = self.trainloader if train else self.testloader
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def compute_accuracy_ensemble(self, models_dir, train=False):
        # TODO: untested
        # TODO: add ensembling logits
        loader = self.trainloader if train else self.testloader
        path_models = list_models(models_dir)
        y_pred = np.zeros((loader.dataset.data.shape[0], len(self.classes)))
        with torch.no_grad():
            for path_model in path_models:
                model = guess_and_load_model(path_model, data=self)
                for i_batch, batch in enumerate(loader):
                    inputs = batch[0].to(self.device)
                    probs = torch.nn.functional.softmax(model(inputs), dim=1)
                    y_pred[(i_batch * loader.batch_size):((i_batch + 1) * loader.batch_size), :] += probs.numpy()
        y_pred /= len(path_models)
        label_pred = np.argmax(y_pred, axis=1)
        correct = total = 0
        for i_batch, batch in enumerate(loader):
            labels = batch[1].numpy()
            total += labels.size(0)
            correct += (y_pred[(i_batch * loader.batch_size):((i_batch + 1) * loader.batch_size), :] == labels).sum().item()
        return correct / total, y_pred, label_pred


class MNIST(DataBase):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    def __init__(self, batch_size, num_workers=0):
        self.trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                       shuffle=True, num_workers=num_workers)
        self.testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                      shuffle=False, num_workers=num_workers)
        super().__init__(batch_size=batch_size, num_workers=num_workers)


class CIFAR10(DataBase):

    def __init__(self, batch_size, num_workers=0):
        # normalize = transforms.Normalize(mean=(0.49139968, 0.48215841, 0.44653091),
        #                                  std=(0.24703223, 0.24348513, 0.26158784))
        normalize = transforms.Normalize(mean=(0., 0., 0.),
                                         std=(1., 1., 1.))
        #normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                                 std=(0.5, 0.5, 0.5))
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             normalize])
        self.trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                       shuffle=True, num_workers=num_workers)
        self.testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                      shuffle=False, num_workers=num_workers)
        super().__init__(batch_size=batch_size, num_workers=num_workers)

    def get_input_shape(self):
        return (1, 3, 32, 32)


class CIFAR100(DataBase):

    def __init__(self, batch_size, num_workers=0):
        # normalize = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
        normalize = transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             normalize])
        self.trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                       shuffle=True, num_workers=num_workers)
        self.testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, pin_memory=self.use_cuda,
                                                      shuffle=False, num_workers=num_workers)
        super().__init__(batch_size=batch_size, num_workers=num_workers)


class ImageNet(DataBase):

    def __init__(self, batch_size, path='/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/', num_workers=0):
        # if DATAPATH env var is set, overright default value
        if os.environ.get('DATAPATH', False):
            path = os.environ.get('DATAPATH')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[0., 0., 0.],
                                         std=[1., 1., 1.])
        traindir = os.path.join(path, 'train')
        self.trainset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=self.use_cuda)
        testdir = os.path.join(path, 'validation')
        self.testset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=self.use_cuda)
        self.min_pixel_value = 0
        self.max_pixel_value = 1
        super().__init__(batch_size=batch_size, num_workers=num_workers)

    def get_input_shape(self):
        return (1, 3, 224, 224)
