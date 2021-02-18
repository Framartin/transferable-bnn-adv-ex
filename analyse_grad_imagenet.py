"""
Compute the L2 norm of gradients on CIFAR-10 on all test examples
To analyse the 0-gradient issue
"""

import torch
from utils.data import CIFAR10, CIFAR100, ImageNet
from utils.helpers import guess_and_load_model, load_classifier, list_models
from art.utils import check_and_transform_label_format


#ensemble_path = list_models("models/ImageNet/resnet50/cSGLD_cycles5_samples3_bs256/")  # BNN
ensemble_path = list_models("models/ImageNet/resnet50/deepens_imagenet/") # DNN
data = ImageNet(batch_size=100, path="../data/ILSVRC2012")
model = guess_and_load_model(ensemble_path[0], data=data, load_as_ghost=False,
                             input_diversity=False)
classifier = load_classifier(model, data=data)

tol = 10e-8
sizes_grads = torch.zeros(10000)

for (batch_id, batch_all) in enumerate(data.testloader):
    batch_index_1, batch_index_2 = batch_id * data.batch_size, (batch_id + 1) * data.batch_size
    x = batch_all[0].to(classifier.device)
    y = check_and_transform_label_format(batch_all[1].cpu().numpy(), classifier.nb_classes)
    #y = batch_all[1].to(classifier.device)
    y = torch.from_numpy(y).to(classifier.device)
    grad = classifier.loss_gradient(x=x, y=y)
    ind = tuple(range(1, len(x.shape)))
    norm_grad = torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True))
    sizes_grads[batch_index_1:batch_index_2] = norm_grad.reshape([x.size(0)]).cpu()
    if batch_index_2 >= 10000-1:
        break

(sizes_grads <= tol).sum() / 10000 * 100
(sizes_grads <= 1e-6).sum() / 10000 * 100

# BNN
# tol:  0.2400 %
# 1e-6: 1.0200 %

# DNN
# tol:  0.0800 %
# 1e-6: 0.3800 %