# import torch  # https://pytorch.org

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_MNIST_dloaders(batch_size=64, size=28, path='~/Data',download = True, num_workers=8):
    
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    train_data = datasets.MNIST(path, train=True,
                                download=download, transform=all_transforms)
    test_data = datasets.MNIST(path, train=False,
                                download=download, transform=all_transforms)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    
    return trainloader, testloader


def get_cifar_dloaders(batch_size=64, size=32, path='./data/cifar10_data',download = True, num_workers=8):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(path, train=True, download=download, 
                                    transform=transform_train)
    test_data = datasets.CIFAR10(path, train=False, download=download,
                                    transform=transform_test)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    
    return trainloader, testloader