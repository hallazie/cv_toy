# coding:utf-8
#
# @author:xsh

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import logging
import os

from torch.autograd import Variable
from torch.utils import data as t_data
from torchvision import transforms


os.environ['KMP_DUPLICATE_LIB_OK']='True'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

data_path = '../data'
batch_size = 32
g_steps = 8
d_steps = 16
printing_steps = 50
epochs = 500

class generator(nn.Module):
	def __init__(self, inp=1, out=784):
		super(generator, self).__init__()

		self.net = nn.Sequential(
			nn.ConvTranspose2d(inp, 32, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(inplace=True),
			# nn.Sigmoid(),
			nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True),
			# nn.Sigmoid(),
			nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(inplace=True),
			# nn.Sigmoid(),
			nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(inplace=True),
			# nn.Sigmoid(),
			nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(inplace=True),
			# nn.Sigmoid(),
			nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(1),
			nn.LeakyReLU(inplace=True),
			# nn.Sigmoid()
		)

	def forward(self, x):
		x = self.net(x)
		return x

class discriminator(nn.Module):
	def __init__(self, inp=784, out=1):
		super(discriminator, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(inp, 300),
			nn.LeakyReLU(inplace=True),
			nn.Linear(300, 300),
			nn.LeakyReLU(inplace=True),
			nn.Linear(300, 300),
			nn.LeakyReLU(inplace=True),
			nn.Linear(300, 200),
			nn.LeakyReLU(inplace=True),
			nn.Linear(200, out),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.net(x)
		return x

def plot_image(array, number=None):
	array = array.cpu().detach()
	a1, a2, a3, a4 = array[0].reshape(28,28), array[1].reshape(28,28), array[2].reshape(28,28), array[3].reshape(28,28)
	plt.subplot(221)
	plt.imshow(a1, cmap='binary')
	plt.subplot(222)
	plt.imshow(a2, cmap='binary')
	plt.subplot(223)
	plt.imshow(a3, cmap='binary')
	plt.subplot(224)
	plt.imshow(a4, cmap='binary')
	if number:
		plt.xlabel(number, fontsize='x-large')
	plt.show()

def make_some_noise():
	return torch.rand(batch_size, 100)

def make_some_noise_2d():
	return torch.rand(batch_size, 1, 7, 7)

def run():
	if torch.cuda.is_available():
		logger.info('using GPU for training')
		dis = discriminator()
		gen = generator()
	else:
		logger.info('using CPU for training')
		dis = discriminator()
		gen = generator()

	data_transforms = transforms.Compose([transforms.ToTensor()])
	mnist_trainset = datasets.MNIST(root=data_path, train=True, download=False, transform=data_transforms)
	dataloader_mnist_train = t_data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
	logger.info('data init finished')

	# criterion1 = nn.BCELoss()
	# optimizer1 = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)
	# criterion2 = nn.BCELoss()
	# optimizer2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

	criterion1 = nn.BCELoss()
	optimizer1 = optim.Adam(dis.parameters(), lr=1e-5)
	criterion2 = nn.BCELoss()
	optimizer2 = optim.Adam(gen.parameters(), lr=1e-5)

	for epoch in range(epochs):
		dis.zero_grad()
		for d_step in range(d_steps):
			for inp_real, _ in dataloader_mnist_train:
				inp_real_x = inp_real
				break
			inp_real_x = inp_real_x.reshape(batch_size, 784)
			dis_real_out = dis(inp_real_x)
			dis_real_loss = criterion1(dis_real_out, Variable(torch.ones(batch_size, 1)))
			inp_fake_x_gen = make_some_noise_2d()
			inp_fake_x = gen(inp_fake_x_gen).detach()
			dis_fake_out = dis(inp_fake_x.reshape(batch_size, 784))
			dis_fake_loss = criterion1(dis_fake_out, Variable(torch.zeros(batch_size, 1)))
			dis_total_loss = (dis_real_loss + dis_fake_loss) / 2.
			dis_total_loss.backward()
			optimizer1.step()
		gen.zero_grad()
		for g_step in range(g_steps):
			gen_inp = make_some_noise_2d()
			gen_out = gen(gen_inp)
			dis_out_gen_training = dis(gen_out.reshape(batch_size, 784))
			gen_loss = criterion2(dis_out_gen_training, Variable(torch.ones(batch_size, 1)))
			gen_loss.backward()
			optimizer2.step()
		if (epoch+1)%printing_steps==0:
			torch.save(dis, '../data/dis-%s.pkl' % (epoch + 1))
			torch.save(gen, '../data/gen-%s.pkl' % (epoch + 1))
		logger.info(f'step {epoch} finished, with G-Loss: {gen_loss.data.item()}, D-Loss: {dis_fake_loss.data.item()} + {dis_real_loss.data.item()}')

def test():
	gen = torch.load('../data/gen-%s.pkl' % 400)
	gen_inp = make_some_noise_2d()
	gen_out = gen(gen_inp)
	plot_image(gen_out)

if __name__ == '__main__':
	run()


