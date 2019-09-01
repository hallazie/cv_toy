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
	discriminator = Discriminator(256, 192, BATCH_SIZE).to(device)
	dataset = SaliconSet(DATA_PATH, LABEL_PATH)
	rawloader = DataLoader(
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
		# first train dis with groundtruth, backward, train dis with fake, backward
		# then gen salmap, get dis result, calc GAN loss, backward.
		avg_loss_dis, avg_loss_gen = [], []
		dataloader = iter(rawloader)
		try:
			for s in range(DIS_STEPS):
				discriminator.zero_grad()
				data_batch, label_batch  = dataloader.__next__()
				input_batch = Variable(data_batch).to(device)
				real_batch = torch.cat((data_batch, label_batch), 1)
				real_batch = Variable(real_batch).to(device)
				output_batch_real = discriminator(real_batch.float())
				loss_content_real = criterion_dis(output_batch_real, Variable(torch.ones(BATCH_SIZE, 1)).to(device))
				avg_loss_dis.append(loss_content_real.data.item())
				loss_content_real.backward()
				gen_batch = generator(input_batch).detach()
				input_fake_batch = torch.cat((input_batch, gen_batch), 1)
				output_batch_fake = discriminator(input_fake_batch)
				loss_content_fake = criterion_dis(output_batch_fake, Variable(torch.zeros(BATCH_SIZE, 1)).to(device))
				avg_loss_dis.append(loss_content_fake.data.item())
				loss_content_fake.backward()
				optimizer_dis.step()
			logger.info('epoch %s, dis-loss=%s' % (e, float(sum(avg_loss_dis))/len(avg_loss_dis)))
			for g in range(GEN_STEPS):
				generator.zero_grad()
				data_batch, _  = dataloader.__next__()
				input_batch = Variable(data_batch).to(device)
				gen_output = generator(input_batch).detach()
				input_fake_batch = torch.cat((input_batch, gen_batch), 1)
				output_batch = discriminator(input_fake_batch)
				loss_adversarial = criterion_gen(output_batch, Variable(torch.ones(BATCH_SIZE, 1)).to(device))
				loss_content = criterion_dis(output_batch, Variable(torch.zeros(BATCH_SIZE, 1)).to(device))
				loss_total = loss_adversarial + ALPHA * loss_content
				avg_loss_gen.append(loss_total.data.item())
				loss_total.backward()
				optimizer_dis.step()
				optimizer_gen.step()
			logger.info('epoch %s, gen-loss=%s' % (e, float(sum(avg_loss_gen))/len(avg_loss_gen)))
		except StopIteration as stoperror:
			pass
		finally:
			if e % SAVE_STEP==0 and e!=0:
				torch.save(generator, os.path.join(PARAM_PATH, 'generator.%s.pkl' % (e)))
				torch.save(discriminator, os.path.join(PARAM_PATH, 'discriminator.%s.pkl' % (e)))

if __name__ == '__main__':
	# get_logger()
	train()