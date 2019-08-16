# coding:utf-8
#
# @author:xsh

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets

from torch.autograd import Variable
from torch.utils import data as t_data
from torchvision import transforms

data_path = '/Users/xiaoshanghua/Data/cv'
batch_size = 4
g_steps = 100
d_steps = 100
printing_steps = 200
epochs = 50

class generator(nn.Module):
	def __init__(self, inp, out):
		super(generator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(inp, 300),
			nn.ReLU(inplace=True),
			nn.Linear(300, 1000),
			nn.ReLU(inplace=True),
			nn.Linear(1000, 800),
			nn.ReLU(inplace=True),
			nn.Linear(800, out)
		)

	def forward(self, x):
		x = self.net(x)
		return x

class discriminator(nn.Module):
	def __init__(self, inp, out):
		super(discriminator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(inp, 300),
			nn.ReLU(inplace=True),
			nn.Linear(300, 300),
			nn.ReLU(inplace=True),
			nn.Linear(300, 200),
			nn.ReLU(inplace=True),
			nn.Linear(200, out),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.net(x)
		return x

def plot_image(array, number=None):
	array = array.detach()
	array = array.reshape(28, 28)
	plt.imshow(array, cmap='binary')
	plt.xticks([])
	plt.yticks([])
	if number:
		plt.xlabel(number, fontsize='x-large')
	plt.show()

def make_some_noise():
	return torch.rand(batch_size, 100)

def run():
	data_transforms = transforms.Compose([transforms.ToTensor()])
	mnist_trainset = datasets.MNIST(root=data_path, train=True, download=True, transforms=data_transforms)
	dataloader_mnist_train = t_data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
	
	dis = discriminator()
	gen = generator()
	criterion1 = nn.BCELoss()
	optimizer1 = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)
	criterion2 = nn.BCELoss()
	optimizer2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(epochs):
		print(epoch)
		for d_step in range(d_steps):
			dis.zero_grad()
			for inp_real, _ in dataloader_mnist_train:
				inp_real_x = inp_real
				break
			inp_real_x = inp_real_x.reshape(batch_size, 784)
			dis_real_out = dis(inp_real_x)
			dis_real_loss = criterion1(dis_real_out, Variable(torch.ones(batch_size, 1)))
			dis_real_loss.backward()
			inp_fake_x_gen = make_some_noise()
			inp_fake_x = gen(inp_fake_x_gen).detach()
			dis_fake_out = dis(inp_fake_x)
			dis_fake_loss = criterion1(dis_fake_out, Variable(torch.zeros(batch_size, 1)))
			dis_fake_loss.backward()
			optimizer1.step()
		for g_step in range(g_steps):
			gen.zero_grad()
			gen_inp = make_some_noise()
			gen_out = gen(gen_inp)
			dis_out_gen_training = dis(gen_out)
			gen_loss = criterion2(dis_out_gen_training, Variable(torch.ones(batch_size, 1)))
			gen_loss.backward()
			optimizer2.step()
		if epoch%printing_steps==0:
			plot_image(gen_out[0])
			plot_image(gen_out[1])
			plot_image(gen_out[2])
			plot_image(gen_out[3])
			print('\n\n')

if __name__ == '__main__':
	run()


