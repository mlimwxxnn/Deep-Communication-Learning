"""
    Load each data set required for the experiment. Data enhancement for images is done here.
"""

import torchvision
import torchvision.transforms as transforms
import os


def load_dataset(dataset, nn_type):
    """
     Get the image data and wrap it as a DataSet type, return the image information of the
     requested data set.
    """
    # Pre-processing for images
    if dataset.startswith('CIFAR'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if nn_type.startswith('Inception'):
            transform_train = transforms.Compose([
                transforms.Resize(299),
                transforms.RandomCrop(299, padding=35),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(32),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(32),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        if nn_type.startswith('Inception'):
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(299),
                                                  transforms.Normalize((0.1307,), (0.3081,))])
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(299),
                                                 transforms.Normalize((0.1307,), (0.3081,))])

    if dataset == 'CIFAR-10':
        if os.path.exists('./data/cifar-10-batches-py'):
            download = False
        else:
            download = True
        num_classes = 10
        in_channels = 3
        img_size = 32
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=download, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=download, transform=transform_test)
    elif dataset == 'CIFAR-100':
        if os.path.exists('./data/cifar-100-python'):
            download = False
        else:
            download = True
        num_classes = 100
        in_channels = 3
        img_size = 32
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=download, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=download, transform=transform_test)
    elif dataset == 'Fashion-MNIST':
        num_classes = 10
        in_channels = 1
        img_size = 28
        trainset = torchvision.datasets.FashionMNIST('./data/', train=True, download=True,
                                                     transform=transform_train)
        testset = torchvision.datasets.FashionMNIST('./data/', train=False, download=True,
                                                    transform=transform_train)
    else:
        raise ValueError(f'dataset not found: [{dataset}]')
    return num_classes, trainset, testset, in_channels, img_size

