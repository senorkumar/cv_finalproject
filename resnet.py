import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#Resnet architechture with 
class ResNet50(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        #declare layers in the network
        self.upsample = nn.Upsample(scale_factor=7)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    
    def forward(self, x):
        print(x.shape)
        x = self.upsample(x)
        x = self.resnet(x)
        return x
