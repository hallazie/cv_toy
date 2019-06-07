# coding:utf-8
# @author: xsh

import torch
import torch.nn as nn
import torch.nn.functional as F

INPLACE = True

class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, exp):
        super(InvertedResidual, self).__init__()
        self.use_res = c_in == c_out
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, exp, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(exp),
            nn.ReLU6(inplace=INPLACE),
            nn.Conv2d(exp, exp, kernel_size=3, stride=1, padding=1, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            nn.ReLU6(inplace=INPLACE),
            nn.Conv2d(exp, c_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fine_net = self.TinyNet()
        self.coarse_net = self.TinyNet()
        self.loss = self.MAELoss()

    def forward(self, fine_batch, coarse_batch, label_batch):
        self.fine_batch = fine_batch
        self.coarse_batch = coarse_batch
        self.label_batch = label_batch
        fine = self.fine_net(self.fine_batch)
        coarse = self.coarse_net(self.coarse_batch)
        # mul = fine
        mul = coarse * fine
        out = self.loss(mul, self.label_batch)
        return out, mul, fine, coarse

    # def SubNet(self):
    #     return nn.Sequential(
    #         self.Conv(3, 32, 3, 1),
    #         self.Pool('max'),
    #         self.ConvDWS(32, 16, 64),
    #         self.ConvDWS(16, 16, 64),
    #         self.Pool('max'),
    #         self.ConvDWS(16, 24, 96),
    #         self.ConvDWS(24, 24, 96),
    #         self.ConvDWS(24, 24, 96),
    #         self.Pool('max'),
    #         self.ConvDWS(24, 32, 128),
    #         self.ConvDWS(32, 32, 128),
    #         self.ConvDWS(32, 32, 128),
    #         self.ConvDWS(32, 32, 128),
    #         self.ConvDWS(32, 64, 256),
    #         self.ConvDWS(64, 64, 256),
    #         self.ConvDWS(64, 128, 512),
    #         self.Conv(128, 1, 1, 0)
    #     )

    def SubNet(self):
        return nn.Sequential(
            self.Conv(3, 32, 3, 1),
            self.Pool('max'),
            InvertedResidual(32, 16, 64),
            InvertedResidual(16, 16, 64),
            self.Pool('max'),
            InvertedResidual(16, 24, 96),
            InvertedResidual(24, 24, 96),
            InvertedResidual(24, 24, 96),
            self.Pool('max'),
            InvertedResidual(24, 32, 128),
            InvertedResidual(32, 32, 128),
            InvertedResidual(32, 32, 128),
            InvertedResidual(32, 32, 128),
            InvertedResidual(32, 64, 256),
            InvertedResidual(64, 64, 256),
            InvertedResidual(64, 128, 512),
            self.Conv(128, 1, 1, 0)
        )

    def OverfitNet(self):
        return nn.Sequential(
            self.Conv(3, 32, 3, 1),
            self.Pool('max'),
            self.Conv(32, 16, 3, 1),
            self.Conv(16, 16, 3, 1),
            self.Pool('max'),
            self.Conv(16, 24, 3, 1),
            self.Conv(24, 24, 3, 1),
            self.Conv(24, 24, 3, 1),
            self.Pool('max'),
            self.Conv(24, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 64, 3, 1),
            self.Conv(64, 64, 3, 1),
            self.Conv(64, 128, 3, 1),
            self.Conv(128, 1, 1, 0)            
        )

    def TinyNet(self):
        return nn.Sequential(
            self.Conv(3, 32, 3, 1),
            self.Pool('max'),
            self.Conv(32, 16, 3, 1),
            self.Conv(16, 16, 3, 1),
            self.Pool('max'),
            self.Conv(16, 24, 3, 1),
            self.Conv(24, 24, 3, 1),
            self.Conv(24, 24, 3, 1),
            self.Pool('max'),
            self.Conv(24, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 64, 3, 1),
            self.Conv(64, 64, 3, 1),
            self.Conv(64, 128, 3, 1),
            self.Conv(128, 1, 1, 0)            
        )

    def MicroNet(self):
        return nn.Sequential(
            self.Conv(3, 32, 3, 1),
            self.Pool('max'),
            self.Conv(32, 16, 3, 1),
            self.Conv(16, 16, 3, 1),
            self.Pool('max'),
            self.Conv(16, 24, 3, 1),
            self.Conv(24, 24, 3, 1),
            self.Conv(24, 24, 3, 1),
            self.Pool('max'),
            self.Conv(24, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 32, 3, 1),
            self.Conv(32, 1, 1, 0)            
        )

    def ConvDWS(self, c_in, c_out, exp):
        return nn.Sequential(
            nn.Conv2d(c_in, exp, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(exp),
            nn.ReLU6(inplace=INPLACE),
            nn.Conv2d(exp, exp, kernel_size=3, stride=1, padding=1, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            nn.ReLU6(inplace=INPLACE),
            nn.Conv2d(exp, c_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(c_out),
            # nn.ReLU6(inplace=INPLACE)
        )

    def Conv(self, c_in, c_out, ksize, pad):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=INPLACE)
        )

    def Pool(self, typ):
        if typ == 'max':
            return nn.MaxPool2d(2, stride=2)

    def MAELoss(self):
        return nn.L1Loss()