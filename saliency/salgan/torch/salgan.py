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
	generator = Generator(640, 480, BATCH_SIZE).to(device)
	discriminator = Discriminator(640, 480, BATCH_SIZE).to(device)
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
			# first train dis with groundtruth, backward, train dis with fake, backward
			# then gen salmap, get dis result, calc GAN loss, backward.
			pass
		if e % SAVE_STEP==0 and e!=0:
			torch.save(model, os.path.join(PARAM_PATH, 'model_%s.pkl' % (e)))

if __name__ == '__main__':
	# get_logger()
	train()