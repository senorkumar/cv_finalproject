from resnet import ResNet50
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = ResNet50(batch_size=1).to(device)

input_image = Image.open('airplane1.png')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
print(input_batch.shape)

output = model(input_batch)
print(output.shape)