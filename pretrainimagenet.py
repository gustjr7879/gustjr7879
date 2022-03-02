import torch.nn as nn
import pretrainedmodels

model = pretrainedmodels.__dict__['fbresnet152'](num_classes=1000, pretrained='imagenet')

model.eval()
