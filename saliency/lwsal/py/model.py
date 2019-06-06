# coding:utf-8
# @author: xsh

import torch
import torch.nn as nn
import torch.nn.functional as F

INPLACE = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fine_net = self.SubNet()
        self.coarse_net = self.SubNet()
        self.loss = self.MAELoss()

    def forward(self, fine_batch, coarse_batch, label_batch):
        self.fine_batch = fine_batch
        self.coarse_batch = coarse_batch
        self.label_batch = label_batch
        fine = self.fine_net(self.fine_batch)
        coarse = self.coarse_net(self.coarse_batch)
        mul = fine * coarse
        out = self.loss(mul, self.label_batch)
        return out, mul, fine, coarse

    def SubNet(self):
        return nn.Sequential(
            self.Conv(3, 32, 3, 1),
            self.Pool('max'),
            self.ConvDWS(32, 16, 64),
            self.ConvDWS(16, 16, 64),
            self.Pool('max'),
            self.ConvDWS(16, 24, 96),
            self.ConvDWS(24, 24, 96),
            self.ConvDWS(24, 24, 96),
            self.Pool('max'),
            self.ConvDWS(24, 32, 128),
            self.ConvDWS(32, 32, 128),
            self.ConvDWS(32, 32, 128),
            self.ConvDWS(32, 32, 128),
            self.ConvDWS(32, 64, 256),
            self.ConvDWS(64, 64, 256),
            self.ConvDWS(64, 128, 512),
            self.Conv(128, 1, 1, 0)
        )

    def ConvDWS(self, c_in, c_out, exp):
        return nn.Sequential(
            nn.Conv2d(c_in, exp, 1, groups=1, padding=0),
            nn.BatchNorm2d(exp),
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(exp, exp, 3, groups=exp, padding=1),
            nn.BatchNorm2d(exp),
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(exp, c_out, 1, groups=1, padding=0),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=INPLACE)
        )

    def Conv(self, c_in, c_out, ksize, pad):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=1, padding=pad),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=INPLACE)
        )

    def Pool(self, typ):
        if typ == 'max':
            return nn.MaxPool2d(2, stride=2)

    def MAELoss(self):
        return nn.L1Loss()