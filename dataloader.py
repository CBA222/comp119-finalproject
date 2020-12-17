"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

#from dataloader.datasets import get_cifar_anomaly_dataset
#from dataloader.datasets import get_mnist_anomaly_dataset
from dataloader.kdd_dataset import *

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

##
def load_data(opt):

    train_ds = KDD_dataset(opt, mode='train')
    valid_ds = KDD_dataset(opt, mode='test')

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

