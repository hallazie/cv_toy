# coding:utf-8
# @author: xsh

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *

FINE_PATH = ''
COARSE_PATH = ''
LABEL_PATH = ''
BATCH_SIZE = 16
EPOCHES = 1000

def train():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Net().to(device)
	dataset = SaliconSet(FINE_PATH, COARSE_PATH, LABEL_PATH)
	dataloader = DataLoader(
		dataset,
		batch_size = BATCH_SIZE,
		shuffle = True,
		pin_memory = True,
	)
	optimizer = optim.Adam(model.parameters())

	for e in range(EPOCHES):
		model.train()
		for batch_i, (_, fines, coarses, labels) in enumerate(dataloader):
			batch_done = len(dataloader) * e + batch_i
			fines = Variable(fines.to(device))
			coarses = Variable(coarses.to(device))
			labels = Variable(labels.to(device), requires_grad=False)
			loss, outputs = model(fines, coarses, labels)


if __name__ == '__main__':
	train()