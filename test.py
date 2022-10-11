
import torch
import DiffConv
from DiffConv import DiffConv
from numpy import genfromtxt
import numpy as np
import torch.nn as nn

import torch.nn.functional as F

from os import listdir
from os.path import isfile, join


from torch.optim.lr_scheduler import MultiStepLR
import os
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import math

import time
import sys
from torchvision.utils import save_image
from multiprocessing import set_start_method



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.con1 = nn.Conv2d(3, 128, 11,4)
        self.pool1 = nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(128, 256, 5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.con3 = nn.Conv2d(256,512, 5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.con4 = nn.Conv2d(2560,1024, 4)
        self.diffcon = DiffConv()
        
        self.linear2= nn.Linear(1024,4096)


    def forward(self, x):
        x = F.relu(self.con1(x))
        x = self.pool1(x)
        x = F.relu(self.con2(x))
        x = self.pool2(x)
        x = F.relu(self.con3(x))
        x = self.pool3(x)
        x = self.diffcon(x)
        
        x = F.relu(self.con4(x))

        x = x.view(x.shape[0],-1)
        x = self.linear2(x)
        


        
        return x



model = Net()

#model.forward(torch.rand(5,3,256,256))

p = DiffConv()
x= torch.rand(1,1,5,5);

print(x[0][0])

t = p.forward(x);
print(t.size())

print(t[0][0])
print(t[0][1])
print(t[0][2])
print(t[0][3])
print(t[0][4])




