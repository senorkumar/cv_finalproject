import os
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#Resnet architechture modifed for use on CIFAR10
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        #declare layers in the network
        #upsample image by 7 from bsx3x32x32 to bsx3x224x224
        self.upsample = nn.Upsample(scale_factor=7)
        #we are using an untrained resnet50, so load it in
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        #we are classifying on cifar10 which has 10 output classes not 1000, so fix this
        self.resnet.fc = nn.Linear(in_features=2048, out_features=10)

        #convert activations into confidence
        self.softmax = nn.Softmax(dim=1)

    
    #run input through each layer of network in order
    def forward(self, x):
        x = self.upsample(x)
        x = self.resnet(x)
        x = self.softmax(x)
        return x

