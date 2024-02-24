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
    return device


def create_dataset():

    # Preprocessing-function
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
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


def split_dataset(train_ds, test_ds, device):
    # training indices that will be used for validation

    # Percentage of validation split
    valid_size = 0.2

    num_train = len(train_ds)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index = indices[split:]
    valid_index = indices[:split]

    # samplers for random subset sampling
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    print(f"Train dataset has: {len(train_ds)} images, which are split into:\n"
          f" {len(train_index)} train samples and\n"
          f" {len(valid_index)} validation samples.")
    print(f"Test dataset has {len(test_ds)} images.\n")

    # wrap datasets into iterable datasets using DataLoader

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, sampler=train_sampler, num_workers=4,
                                               pin_memory=True, pin_memory_device=device)
    valid_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, sampler=valid_sampler, num_workers=4,
                                               pin_memory=True, pin_memory_device=device)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=3, shuffle=False, num_workers=1,
                                              pin_memory=True, pin_memory_device=device)

    loaders = dict(train=train_loader, valid=valid_loader, test=test_loader)
    return loaders


# Visualize datasets
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # (1 is height, 2 is width, 0 is batch size)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    device_name = device_check()
    train, test, classnames = create_dataset()
    dictionary = split_dataset(train, test, device_name)

    # Get a batch of training data
    image, classes = next(iter(dictionary['train']))
    image = image[:8]
    classes = classes[:8]

    # Make a grid from batch
    out = torchvision.utils.make_grid(image, nrow=4)
    imshow(out, title=[classnames[x] for x in classes])
