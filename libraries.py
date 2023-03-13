import numpy as np                              # for transformations and numpy conversions
import matplotlib.pyplot as plt                 # for visualizing data
import cv2 as cv                                # for visualizing data
import torch                                    # torch package
import torchvision                              # to download datasets and make use of dataloadera
import torchvision.transforms as transforms     # to make transformations
import torch.nn as nn                           # the CNN class
import torch.nn.functional as F                 # convolution function (ReLU)
import torch.optim as opt                       # optimizer function (SGD, Adam, Adagrad etc)

if torch.cuda.is_available():                   # checking GPU compatibility
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(dev)
