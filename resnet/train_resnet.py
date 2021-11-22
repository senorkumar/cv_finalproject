from resnet import ResNet50
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True)
