# coding:utf-8
# @author: xsh

import torch
import torch.optim as optim

import logging

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *

logger_name = 'train_logger'
logger = logging.getLogger(logger_name)

FINE_PATH = 'E:/data/saliency/SALICON/Crop/fine'
COARSE_PATH = 'E:/data/saliency/SALICON/Crop/coarse'
LABEL_PATH = 'E:/data/saliency/SALICON/Crop/label'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/params'
BATCH_SIZE = 128
EPOCHES = 1000
GRAD_ACCUM = 1

def get_logger():
	logger.setLevel(logging.INFO)
	log_path = './log.log'
	filehandle = logging.FileHandler(log_path)
	filehandle.setLevel(logging.INFO)
	fmt = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s'
	date_fmt = '%a %d %b %Y %H:%M:%S'
	formatter = logging.Formatter(fmt, date_fmt)
	filehandle.setFormatter(formatter)
	logger.addHandler(filehandle)


def train():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Net().to(device)
	# model = torch.load(os.path.join(PARAM_PATH, 'model_1.pkl')).to(device)
	dataset = SaliconSet(FINE_PATH, COARSE_PATH, LABEL_PATH)
	dataloader = DataLoader(
		dataset,
		batch_size = BATCH_SIZE,
		shuffle = True,
		pin_memory = True,
	)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	for e in range(EPOCHES):
		model.train()
		for batch_i, (fines, coarses, labels) in enumerate(dataloader):
			batch_done = len(dataloader) * e + batch_i
			fines = Variable(fines.to(device))
			coarses = Variable(coarses.to(device))
			labels = Variable(labels.to(device), requires_grad=False)
			loss, _, _, _ = model(fines, coarses, labels)
			print('[INFO] epoch %s, batch %s, MAELoss = %s' % (e, batch_i, loss.data.item()))
			logger.info('epoch %s, batch %s, MAELoss = %s' % (e, batch_i, loss.data.item()))
			loss.backward()
			# if batch_done % GRAD_ACCUM == 0:
			if True:
				optimizer.step()
				optimizer.zero_grad()
		# if e % 1 == 0:
		if True:
			torch.save(model, os.path.join(PARAM_PATH, 'model_%s.pkl' % (e + 1)))


if __name__ == '__main__':
	get_logger()
	train()