import torch.nn as nn
from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
import pretrainedmodels
import json
model = wrn28_10_cifar10()

# 1. generate hierarchy from pretrained model
generate_hierarchy(dataset='CIFAR10', arch='wrn28_10_cifar10', model=model, method='random',pretrained = True)

# 2. Fine-tune model with tree supervision loss
criterion = nn.CrossEntropyLoss()
criterion = SoftTreeSupLoss(dataset='CIFAR10', hierarchy='induced-wrn28_10_cifar10', criterion=criterion)

# 3. Run inference using embedded decision rules
model = SoftNBDT(model=model, dataset='CIFAR10', hierarchy='induced-wrn28_10_cifar10')


