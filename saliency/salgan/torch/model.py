# coding:utf-8
# 
# @author: xsh

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2

from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlattenOperator(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def Conv(c_in, c_out, kernel_size, padding, dilation=1):
	return nn.Sequential(
		nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
		nn.BatchNorm2d(c_out),
		nn.ReLU6(inplace=True)
	)

def Upsample(scale_factor=2):
	return nn.Upsample(scale_factor=scale_factor, mode='bilinear')

def MaxPool():
	return nn.MaxPool2d(2, stride=2, padding=0)

def Flatten():
	return FlattenOperator()

def Linear(inp_size, num_filter, activation):
	return nn.Sequential(
		nn.Linear(inp_size, num_filter),
		nn.BatchNorm2d(1),
		nn.Sigmoid() if activation=='sigmoid' else nn.Tanh(),
		)

def Backbone(layer_define):
	seq = []
	for layer in layer_define:
		if layer[0] == 'c':
			seq.append(Conv(*layer[1:]))
		elif layer[0] == 'p':
			seq.append(MaxPool())
		elif layer[0] == 'u':
			seq.append(Upsample(layer[1]))
		elif layer[0] == 'f':
			seq.append(Flatten())
		elif layer[0] == 'l':
			seq.append(Linear(*layer[1:]))
	return nn.Sequential(*seq)

class Discriminator(nn.Module):
	def __init__(self, inputwidth, inputheight, batchsize):
		self.layer_define = [
			('c', 4, 3, 1, 1, 1),
			('c', 3, 32, 3, 1, 1),
			('p'),
			('c', 32, 64, 3, 1, 1),
			('c', 64, 64, 3, 1, 1),
			('p'),
			('c', 64, 64, 3, 1, 1),
			('c', 64, 64, 3, 1, 1),
			('p'),
			('f'),
			('l', 100, 'tanh'),
			('l', 2, 'tanh'),
			('l', 1, 'sigmoid')
		]
		self.backbone = Backbone(self.layer_define)

	def forward(self, batch_data):
		x = self.backbone(batch_data)
		return x

class Generator(nn.Module):
	def __init__(self, inputwidth, inputheight, batchsize):
		super(Model, self).__init__()
		self.layer_define = [
			('c', 3, 64, 3, 1, 1),
			('p'),
			('c', 64, 128, 3, 1, 1),
			('p'),
			('c', 128, 256, 3, 1, 1),
			('p'),
			('c', 256, 512, 3, 1, 1),
			('p'),
			('c', 512, 512, 3, 1, 1),
			('c', 512, 512, 3, 1, 1),
			('u'),
			('c', 512, 256, 3, 1, 1),
			('u'),
			('c', 256, 128, 3, 1, 1),
			('u'),
			('c', 128, 64, 3, 1, 1),
			('u'),
			('c', 64, 1, 3, 1, 1),
		]
		self.backbone = Backbone(self.layer_define)

	def forward(self, batch_data):
		x = self.backbone(batch_data)
		return x

if __name__ == '__main__':
	pass
