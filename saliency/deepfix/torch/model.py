'''
Implementation of MLNet model by Cornia et al.

@author: Xiao Shanghua
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2

from torchvision.models import vgg16
from scipy.ndimage.filters import gaussian_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_bias(inputwidth, inputheight, batchsize):
    bias = np.zeros((1, 16, inputwidth//16, inputheight//16))
    for i in range(1,5):
        for j in range(1,5):
            img = np.zeros((256, 256)).astype(np.float32)
            img[127, 127] = 255.
            img[127, 128] = 255.
            img[128, 127] = 255.
            img[128, 128] = 255.
            blur = gaussian_filter(img, sigma=(i*6, j*6))
            blur = 255. * (blur - np.min(blur)) / (np.max(blur) - np.min(blur))
            bias[0,i,:] = cv2.resize(blur, (inputwidth//16, inputheight//16), interpolation=cv2.INTER_LANCZOS4).transpose()
    
    return torch.tensor(bias.repeat(batchsize, 0)).float().to(device)

class InceptionModule(nn.Module):
    def __init__(self, c_in, c_out):
        super(InceptionModule, self).__init__()
        self.seq_1 = nn.Sequential(
            self.Conv(c_in, c_in//4, 3, 1)
        )
        self.seq_2 = nn.Sequential(
            self.Conv(c_in, c_in//4, 1, 0),
            self.Conv(c_in//4, c_in//2, 3, 1)
        )
        self.seq_3 = nn.Sequential(
            self.Conv(c_in, c_in//16, 1, 0),
            self.Conv(c_in//16, c_in//8, 3, 2, 2)
        )
        self.seq_4 = nn.Sequential(
            self.MaxPool(3, 1, 1),
            self.Conv(c_in, c_in//8, 1, 0)
        )

    def forward(self, x):
        seq_1 = self.seq_1(x)
        seq_2 = self.seq_2(x)
        seq_3 = self.seq_3(x)
        seq_4 = self.seq_4(x)
        return torch.cat([seq_1, seq_2, seq_3, seq_4], 1)

    def Conv(self, c_in, c_out, kernel_size, padding, dilation=1):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True)
        )

    def MaxPool(self, kernel_size=3, stride=2, padding=0):
        return nn.MaxPool2d(3, stride=stride, padding=padding)

class Model(nn.Module):
    def __init__(self, inputwidth=640, inputheight=480, batchsize=1, use_pretrained=False):
        super(Model, self).__init__()
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            self.features = nn.ModuleList(list(vgg16(pretrained=True).features)[:-1])
            self.layer_define = [
                ('i', 512, 512),
                ('i', 512, 512),
            ]
        else:
            self.layer_define = [
                ('c', 3, 64, 3, 1, 1),
                ('c', 64, 64, 3, 1, 1),
                ('p', 3, 2),
                ('c', 64, 128, 3, 1, 1),
                ('c', 128, 128, 3, 1, 1),
                ('p', 3, 2),
                ('c', 128, 256, 3, 1, 1),
                ('c', 256, 256, 3, 1, 1),
                ('c', 256, 256, 3, 1, 1),
                ('p', 3, 2),
                ('c', 256, 512, 3, 1, 1),
                ('c', 512, 512, 3, 1, 1),
                ('c', 512, 512, 3, 1, 1),
                ('p', 3, 1),
                ('i', 512, 512),
                ('i', 512, 512),
            ]
        self.bias = init_bias(inputwidth, inputheight, batchsize)
        self.backbone = self.Backbone()
        self.biasconv1 = self.Conv(528, 512, 5, 12, 6)
        self.biasconv2 = self.Conv(528, 512, 5, 12, 6)
        self.output = self.Conv(512, 1, 1, 0)

    def forward(self, x):
        if self.use_pretrained:
            x = self.features(x)
        x = self.backbone(x)
        x = torch.cat([x, self.bias], 1)
        x = self.biasconv1(x)
        x = torch.cat([x, self.bias], 1)
        x = self.biasconv2(x)
        x = self.output(x)
        return x

    def Backbone(self):
        seq = []
        for layer in self.layer_define:
            if layer[0] == 'c':
                seq.append(self.Conv(*layer[1:]))
            elif layer[0] == 'p':
                seq.append(self.MaxPool())
            elif layer[0] == 'i':
                seq.append(InceptionModule(*layer[1:]))
        return nn.Sequential(*seq)

    def Conv(self, c_in, c_out, kernel_size, padding, dilation=1):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True)
        )

    def MaxPool(self):
        return nn.MaxPool2d(2, stride=2)

if __name__ == "__main__":
    data = torch.zeros(1, 3, 480, 640)
    m = Model()
    a = m(data)
    print(a.shape)