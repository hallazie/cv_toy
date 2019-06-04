# coding:utf-8
# @author: xsh

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *

FINE_PATH = 'E:/data/saliency/SALICON/Crop/fine'
COARSE_PATH = 'E:/data/saliency/SALICON/Crop/coarse'
LABEL_PATH = 'E:/data/saliency/SALICON/Crop/label'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/params'
BATCH_SIZE = 16
EPOCHES = 1000
GRAD_ACCUM = 64

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
		for batch_i, (fines, coarses, labels) in enumerate(dataloader):
			batch_done = len(dataloader) * e + batch_i
			fines = Variable(fines.to(device))
			coarses = Variable(coarses.to(device))
			labels = Variable(labels.to(device), requires_grad=False)
			loss, outputs = model(fines, coarses, labels)
			print('[INFO] epoch %s, batch %s, MAELoss = %s' % (e, batch_i, loss.data.item()))
			loss.backward()
			if batch_done % GRAD_ACCUM:
				optimizer.step()
				optimizer.zero_grad()
			if e % 10 == 0:
				# state = {
				# 	'net':model.state_dict(),
				# 	'optimizer':optimizer.state_dict(),
				# 	'epoch':e
				# }
				torch.save(model, os.path.join(PARAM_PATH, 'model_%s.pkl' % e))

if __name__ == '__main__':
	train()