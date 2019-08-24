# coding:utf-8
# 
# @author: xsh

import torch
import torch.optim as optim

import logging

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *
from config import *

# def get_logger():
# 	logger.setLevel(logging.INFO)
# 	log_path = './log.log'
# 	filehandle = logging.FileHandler(log_path)
# 	filehandle.setLevel(logging.INFO)
# 	fmt = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s'
# 	date_fmt = '%a %d %b %Y %H:%M:%S'
# 	formatter = logging.Formatter(fmt, date_fmt)
# 	filehandle.setFormatter(formatter)
# 	logger.addHandler(filehandle)

logger = logging.getLogger(__name__)

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
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	for e in range(EPOCHES):
		model.train()
		for batch_i, (data_batch, label_batch) in enumerate(dataloader):
			batch_done = len(dataloader) * e + batch_i
			data_batch = Variable(data_batch.to(device))
			label_batch = Variable(label_batch.to(device), requires_grad=False)
			loss= model(data_batch.float(), label_batch.float())
			logger.info('epoch %s, batch %s, EucLoss = %s' % (e, batch_i, loss.data.item()))
			loss.backward()
			if GRAD_ACCUM:
				optimizer.step()
				optimizer.zero_grad()
		if e%SAVE_STEP==0 and e!=0:
			torch.save(model, os.path.join(PARAM_PATH, 'model_%s.pkl' % (e + 1)))


if __name__ == '__main__':
	# get_logger()
	train()