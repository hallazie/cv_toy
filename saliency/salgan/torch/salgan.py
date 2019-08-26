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

def resize_gen2dis(tensor):
	return tensor

def train():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	generator = Generator(640, 480, BATCH_SIZE).to(device)
	discriminator = Discriminator(256, 192, BATCH_SIZE).to(device)
	dataset = SaliconSet(DATA_PATH, LABEL_PATH)
	dataloader = DataLoader(
		dataset,
		batch_size = BATCH_SIZE,
		shuffle = True,
		pin_memory = True,
	)
	criterion_dis = nn.BCELoss()
	optimizer_dis = optim.Adam(discriminator.parameters(), lr=1e-3)
	criterion_gen = nn.BCELoss()
	optimizer_gen = optim.Adam(generator.parameters(), lr=1e-3)

	for e in range(EPOCHES):
		model.train()
		# first train dis with groundtruth, backward, train dis with fake, backward
		# then gen salmap, get dis result, calc GAN loss, backward.
		avg_loss_dis, avg_loss_gen = [], []
		for s in range(DIS_STEPS):
			discriminator.zero_grad()
			data_batch, label_batch  = dataloader.__next__()
			real_batch = np.concatenate((data_batch, label_batch), axis=1)
			real_batch = Variable(real_batch.to(device))
			real_batch = resize_gen2dis(real_batch)
			output_batch_real = discriminator(real_batch.float())
			loss_content_real = criterion_dis(output_batch_real, Variable(torch.ones(batch_size, 1)).to(device))
			avg_loss_dis.append(loss_content_real.data.item())
			loss_content_real.backward()
			gen_batch = generator(data_batch).detach()
			gen_batch = resize_gen2dis(gen_batch)
			output_batch_fake = discriminator(gen_batch)
			loss_content_fake = criterion_dis(output_batch_fake, Variable(torch.zeros(batch_size, 1)).to(device))
			avg_loss_dis.append(loss_content_fake.data.item())
			loss_content_fake.backward()
			optimizer_dis.step()
		for s in range(GEN_STEPS):
			generator.zero_grad()
			data_batch, label_batch  = dataloader.__next__()
			gen_output = generator(data_batch).detach()
			gen_output = resize_gen2dis(gen_output)
			output_batch = discriminator(gen_output)
			loss_adversarial = criterion_gen(output_batch, Variable(torch.ones(batch_size, 1)).to(device))
			loss_content = criterion_dis(output_batch, Variable(torch.zeros(batch_size, 1)).to(device))
			loss_total = loss_adversarial + ALPHA * loss_content
			avg_loss_gen.append(loss_total.data.item())
			loss_total.backward()
			optimizer_dis.step()
			optimizer_gen.step()
		logger.info('epoch %s, gen-loss=%s, dis-loss=%s' % (e, float(sum(avg_loss_gen))/len(avg_loss_gen), float(sum(avg_loss_dis))/len(avg_loss_dis)))

		if e % SAVE_STEP==0 and e!=0:
			torch.save(generator, os.path.join(PARAM_PATH, 'generator.%s.pkl' % (e)))
			torch.save(discriminator, os.path.join(PARAM_PATH, 'discriminator.%s.pkl' % (e)))

if __name__ == '__main__':
	# get_logger()
	train()