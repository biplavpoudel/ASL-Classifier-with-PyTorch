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
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        # Add random affine transformation
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Add random perspective transformation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root="./Input/asl_alphabets/train_new", transform=train_transforms)
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

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, sampler=train_sampler, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, shuffle=False, sampler=valid_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, pin_memory=True)

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
