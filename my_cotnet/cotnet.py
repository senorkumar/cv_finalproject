import sys
sys.path.append("..")
import CoTNet.models.cotnet
import os
import torch
from torch import nn
from torch._C import InferredType
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CoTNet50(nn.Module):
    def __init__(self):
        super().__init__()
         #declare layers in the network

        #upsample image by 7 from bsx3x32x32 to bsx3x224x224
        self.upsample = nn.Upsample(scale_factor=7)

        #we are using an untrained cotnet50, so load it in
        self.cotnet = CoTNet.models.cotnet.cotnet50(pretrained=False)


    def forward(self, x):
        x = self.upsample(x)
        x = self.cotnet(x)
        return x



