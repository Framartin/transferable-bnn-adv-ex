import torch
from torch import nn
import torch.nn.functional as F


class MnistFc(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        if pretrained:
            raise NotImplementedError()
        super(MnistFc, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


class MnistCnn(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        if pretrained:
            raise NotImplementedError()
        super(MnistCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


class CifarLeNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(CifarLeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


class TorchEnsemble(nn.Module):

    def __init__(self, models, ensemble_logits=False):
        """
        :param models: list of pytorch models to ensemble
        :param ensemble_logits: True if ensemble logits, False to ensemble probabilities
        :return probablities if ensemble_logits is False, logits if True
        """
        super(TorchEnsemble, self).__init__()
        if len(models) < 1:
            raise ValueError('Empty list of models')
        self.models = nn.ModuleList(models)
        self.ensemble_logits = ensemble_logits
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # clone to make sure x is not changed by inplace methods
        if not self.ensemble_logits:
            x_list = [self.softmax(model(x.clone())) for model in self.models]  # probs
        else:
            x_list = [model(x.clone()) for model in self.models]  # logits
        x = torch.stack(x_list)  # concat on dim 0
        x = torch.mean(x, dim=0, keepdim=False)
        #for model in self.models:
        #    xi = model(x.clone())  # clone to make sure x is not changed by inplace methods

        # x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        # x1 = x1.view(x1.size(0), -1)
        # x2 = self.modelB(x)
        # x2 = x2.view(x2.size(0), -1)
        # x = torch.cat((x1, x2), dim=1)
        # x = self.classifier(F.relu(x))
        return x
