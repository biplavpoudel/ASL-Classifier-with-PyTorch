import torch
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
plt.ion()


def device_check():
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device for training the CNN model.")


def create_dataset():

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

    class_names = test_dataset.classes
    print(f"\nThe class names are:\n {class_names}\n")

    return train_dataset, test_dataset, class_names


def split_dataset(train_dataset, test_dataset):
    # training indices that will be used for validation

    # Percentage of validation split
    valid_size = 0.2

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index = indices[split:]
    valid_index = indices[:split]

    # samplers for random subset sampling
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    print(f"Train dataset has: {len(train_dataset)} images, which are split into:\n"
          f" {len(train_index)} train samples and\n"
          f" {len(valid_index)} validation samples.")
    print(f"Test dataset has {len(test_dataset)} images.\n")

    # wrap datasets into iterable datasets using DataLoader

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, sampler=train_sampler, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, sampler=valid_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=4)

    loaders = dict(train=train_loader, valid=valid_loader, test=test_loader)
    return loaders


# Visualize datasets
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    device_check()
    train, test, class_names = create_dataset()
    dictionary = split_dataset(train, test)

    # print(len(dictionary["train"]))
    # print(len(dictionary["valid"]))
    # print(len(dictionary["test"]))

    # Get a batch of training data
    image, classes = next(iter(dictionary['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(image, padding=2)
    imshow(out, title=[class_names[x] for x in classes])

