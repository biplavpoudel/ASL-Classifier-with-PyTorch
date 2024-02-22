import torch
import os
import numpy as np
from torchvision.datasets import ImageFolder
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler


device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"\nUsing {device} device for training the CNN model.\n")


# Percentage of validation split
valid_size = 0.2

# Preprocessing-function
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = ImageFolder(root="./Input/asl_alphabets/train", transform=train_transforms)
test_dataset = ImageFolder(root="./Input/asl_alphabets/test", transform=test_transforms)

# training indices that will be used for validation
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_index = indices[split:]
valid_index = indices[:split]

# samplers for random subset sampling
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

print(f"Train dataset has: {len(train_dataset)} images, which are split into:\n"
      f" {len(train_index)} train samples and\n"
      f" {len(valid_index)} validation samples.")
print(f"Test dataset has {len(test_dataset)} images.")

# wrap datasets into iterable datasets using DataLoader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=valid_sampler, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

loaders = dict(train=train_loader, valid=valid_loader, test=test_loader)
