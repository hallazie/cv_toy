# coding:utf-8
# 
# @author: xsh

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *
from config import *

def train():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Model(640, 480, BATCH_SIZE).to(device)
	dataset = SaliconSet(DATA_PATH, LABEL_PATH)
	dataloader = DataLoader(
		dataset,
		batch_size = BATCH_SIZE,
		shuffle = True,
		pin_memory = True,
	)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	for e in range(EPOCHES):
		model.train()
		for batch_i, (data_batch, label_batch) in enumerate(dataloader):
			batch_done = len(dataloader) * e + batch_i
			data_batch = Variable(data_batch.to(device))
			label_batch = Variable(label_batch.to(device), requires_grad=False)
			output_batch = model(data_batch.float())
			loss = criterion(output_batch, label_batch.float())
			logger.info('epoch %s, batch %s, EucLoss = %s' % (e, batch_i, loss.data.item()))
			loss.backward()
			if GRAD_ACCUM:
				optimizer.step()
				optimizer.zero_grad()
		if e % SAVE_STEP==0 and e!=0:
			torch.save(model, os.path.join(PARAM_PATH, 'model_%s.pkl' % (e)))

if __name__ == '__main__':
	# get_logger()
	train()