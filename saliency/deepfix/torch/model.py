# coding:utf-8
# 
# @author: xsh

import torch
import torch.nn as nn
import torch.nn.functional as F

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
			self.MaxPool(),
			self.Conv(c_in//8, c_in//8, 1, 0)
		)

	def forward(self, x):
		seq_1 = self.seq_1(x)
		seq_2 = self.seq_2(x)
		seq_3 = self.seq_3(x)
		seq_4 = self.seq_4(x)
		return torch.cat([seq_1, seq_2, seq_3, seq_4], 1)

	def Conv(self, c_in, c_out, kernel_size, padding, dilation=0):
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
			nn.BatchNorm2d(c_out),
			nn.ReLU6(inplace=True)
		)

	def MaxPool(self):
		return nn.MaxPool2d(2, stride=2)

class Model(nn.Module):
	def __init__(self, inputwidth, inputheight, batchsize):
		super(Model, self).__init__()
		self.layer_define = [
			('c', 3, 64, 3, 1, 0),
			('c', 64, 64, 3, 1, 0),
			('p', 0),
			('c', 64, 128, 3, 1, 0),
			('c', 128, 128, 3, 1, 0),
			('p'),
			('c', 128, 256, 3, 1, 0),
			('c', 256, 256, 3, 1, 0),
			('c', 256, 256, 3, 1, 0),
			('p'),
			('c', 256, 512, 3, 1, 0),
			('c', 512, 512, 3, 1, 0),
			('c', 512, 512, 3, 1, 0),
			('p'),
			('i', 512, 512),
			('i', 512, 512),
			('b', 512, 512, 3, 6, 6),
			('b', 512, 512, 3, 6, 6),
			('c', 512, 1, 1, 0, 0)
		]
		self.bias = torch.randn(16, inputwidth//16, inputheight//16, batchsize)
		self.backbone = self.Backbone()
		self.biasconv1 = self.Conv(512, 512, 3,6 ,6)
		self.biasconv2 = self.Conv(512, 512, 3,6 ,6)
		self.loss = self.EucLoss()

	def forward(self, batch_data, batch_label):
		x = self.backbone(batch_data)
		x = self.biasconv1(x)
		x = torch.cat([x, self.bias], 1)
		x = self.biasconv2(x)
		x = torch.cat([x, self.bias], 1)
		x = self.loss(x, batch_label)
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

	def Conv(self, c_in, c_out, kernel_size, padding, dilation=0):
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
			nn.BatchNorm2d(c_out),
			nn.ReLU6(inplace=True)
		)

	def MaxPool(self):
		return nn.MaxPool2d(2, stride=2)

	def EucLoss(self):
		return nn.MSELoss()
