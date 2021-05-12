"""
    Load a variety of neural networks needed for experiments in deep communication learning.

    Note that many of the pre-defined network models in torchvision are more suitable for large images,
    so for relatively small images, changes need to be applied to these pre-defined networks. For example,
    changing the stride of the first convolution layer from 2 to 1, and removing the first MaxPooling from
    the header to avoid losing a lot of information of the image at the beginning. Experiments have shown
    that these small changes to the network parameters can significantly improve the network results.
"""

import torch
import torchvision.models as models
import torch.nn as nn
from models.wideresidual import wideresnet
from models.efficientnet_pytorch import EfficientNet


class FakeMaxPool(nn.Module):
    """
    Create a fake MaxPooling layer to replace the unnecessary MaxPooling layer in the pre-defined
    neural network in torchvision.
    """

    def __init__(self):
        super(FakeMaxPool, self).__init__()

    def forward(self, x):
        return x


def resnet18_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load ResNet-18.
    """
    resnet18 = models.resnet18(pretrained=pretrained)
    if img_size <= 32:
        resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        resnet18.maxpool = FakeMaxPool()
    resnet18.fc = torch.nn.Linear(512, num_classes)
    return resnet18


def mobilenet_v2_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load MobileNet-V2.
    """
    mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
    if img_size <= 32:
        mobilenet_v2.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                bias=False)
    mobilenet_v2.classifier[1] = torch.nn.Linear(1280, num_classes)
    return mobilenet_v2


def wrn_16_4_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load WRN-16-4, note that WRN-16-4 is not pre-trained in the experiment.
    """
    if pretrained:
        raise ValueError('pretrained of WRN-16-4 have to be False!')
    wrn = wideresnet(depth=16, widen_factor=4, num_classes=num_classes, in_channels=in_channels, img_size=img_size)
    return wrn


def inception_v1_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load Inception-V1.
    """
    net = models.googlenet(pretrained=pretrained, init_weights=True)
    if img_size <= 32:
        net.conv1.conv = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        net.maxpool1 = FakeMaxPool()
        net.maxpool2 = FakeMaxPool()
    net.fc = nn.Linear(1024, num_classes)
    return net


def densenet121_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load DenseNet-121.
    """
    net = models.densenet121(pretrained=pretrained)
    if img_size <= 32:
        net.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        net.features[3] = FakeMaxPool()
    net.classifier = nn.Linear(1024, num_classes)
    return net


def resnext50_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load ResNeXt-50.
    """
    net = models.resnext50_32x4d(pretrained=pretrained)
    if img_size <= 32:
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        net.maxpool = FakeMaxPool()
    net.fc = nn.Linear(2048, num_classes)
    return net


def efficientnet_b3_loader(pretrained, num_classes=10, in_channels=3, img_size=32):
    """
    Load EfficientNet-B3, note that EfficientNet-B3 is not pre-trained in the experiment.
    """
    if pretrained:
        raise ValueError('pretrained of EfficientNet-B3 have to be False!')
    model = EfficientNet.from_name('efficientnet-b3', in_channels=in_channels, num_classes=num_classes,
                                       **{'image_size': img_size})
    return model


def get_loader(net_type):
    """
    The contained dictionary is a registry of neural network loaders, this function returns the corresponding
    neural network loader according to the type of network input.
    """
    loader_dic = {
        'ResNet-18': resnet18_loader,
        'WRN-16-4': wrn_16_4_loader,
        'MobileNet-V2': mobilenet_v2_loader,
        'Inception-V1': inception_v1_loader,
        'DenseNet-121': densenet121_loader,
        'ResNeXt-50': resnext50_loader,
        'EfficientNet-B3': efficientnet_b3_loader,
    }
    return loader_dic.get(net_type)
