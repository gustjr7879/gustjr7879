import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

cifar10 = datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Train samples:", train_images.shape, train_labels.shape)
print("Test samples:", test_images.shape, test_labels.shape)

train_images = train_images/255.0
test_images = test_images/255.0
#정규화하여 0,1사이에 분포하도록 배치

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
model.fit(train_images, train_labels, epochs=1)
 
test_loss, test_acc = model.evaluate(test_images, test_labels)
 
print('Test accuracy:', test_acc)



import torch.nn as nn
from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy

generate_hierarchy(dataset='CIFAR10', arch='wrn28_10_cifar10', model=model, method='random',pretrained = True)

criterion = nn.CrossEntropyLoss()

criterion = SoftTreeSupLoss(dataset='CIFAR10', hierarchy='induced-wrn28_10_cifar10', criterion=criterion)


model = SoftNBDT(model=model, dataset='CIFAR10', hierarchy='induced-wrn28_10_cifar10')

