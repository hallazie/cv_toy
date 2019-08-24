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

def init_bias_from_file(inputwidth, inputheight, batchsize):
	bias = np.zeros((1, 16, inputwidth//16, inputheight//16))
	for _,_,fs in os.walk(BIAS_PATH):
		for i, f in enumerate(sorted([x for x in fs if x.endswith('.png')])):
			t = cv2.imread(os.path.join(BIAS_PATH, f), cv2.IMREAD_GRAYSCALE)
			bias[0,i,:] = cv2.resize(cv2.imread(os.path.join(BIAS_PATH, f), cv2.IMREAD_GRAYSCALE), (inputwidth//16, inputheight//16), interpolation=cv2.INTER_LANCZOS4).transpose()
	
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
	def __init__(self, inputwidth, inputheight, batchsize):
		super(Model, self).__init__()
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
		# self.bias = torch.randn(batchsize, 16, inputwidth//16, inputheight//16)
		self.bias = init_bias_from_file(inputwidth, inputheight, batchsize)
		self.backbone = self.Backbone()
		self.biasconv1 = self.Conv(528, 512, 5, 12, 6)
		self.biasconv2 = self.Conv(528, 512, 5, 12, 6)
		self.output = self.Conv(512, 1, 1, 0)
		# self.loss = self.EucLoss()

	def forward(self, batch_data):
		x = self.backbone(batch_data)
		x = torch.cat([x, self.bias], 1)
		x = self.biasconv1(x)
		x = torch.cat([x, self.bias], 1)
		x = self.biasconv2(x)
		x = self.output(x)
		# x = self.loss(x, batch_label)
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

	def EucLoss(self):
		return nn.MSELoss()

if __name__ == '__main__':
	# pass
	init_bias_from_file(640, 480, 1)