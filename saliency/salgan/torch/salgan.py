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
	optimizer_dis = optim.Adagrad(discriminator.parameters(), lr=1e-1, weight_decay=1e-4)
	criterion_gen = nn.MSELoss()
	optimizer_gen = optim.Adagrad(generator.parameters(), lr=1e-1, weight_decay=1e-4)

	for e in range(EPOCHES):
		# first train dis with groundtruth, backward, train dis with fake, backward
		# then gen salmap, get dis result, calc GAN loss, backward.
		avg_loss_dis, avg_loss_gen = [], []
		dataloader = iter(rawloader)
		try:
			if e <= WARMUP:
				for idx, data_batch, label_batch in enumerate(dataloader):
					generator.zero_grad()
					data_batch = Variable(data_batch).to(device)
					label_batch = Variable(label_batch).to(device)
					output_batch = generator(data_batch).detach()
					loss = criterion_gen(output_batch, label_batch)
					avg_loss_gen.append(loss.data.item())
					loss.backward()
					optimizer_gen.step()
				logger.info('epoch %s, warmup-gen-loss=%s' % (e, float(sum(avg_loss_gen))/len(avg_loss_gen)))
			else:
				for s in range(DIS_STEPS):
					discriminator.zero_grad()
					data_batch, label_batch  = dataloader.__next__()
					input_batch = Variable(data_batch).to(device)
					input_real_batch = torch.cat((data_batch, label_batch), 1)
					input_real_batch = Variable(input_real_batch).to(device)
					output_batch_real = discriminator(input_real_batch)
					loss_content_real = criterion_dis(output_batch_real, Variable(torch.ones(BATCH_SIZE, 1)).to(device))
					gen_batch = generator(input_batch).detach()
					input_fake_batch = torch.cat((input_batch, gen_batch), 1)
					input_fake_batch = Variable(input_fake_batch).to(device)
					output_batch_fake = discriminator(input_fake_batch)
					loss_content_fake = criterion_dis(output_batch_fake, Variable(torch.zeros(BATCH_SIZE, 1)).to(device))
					loss_dis = loss_content_real + loss_content_fake
					avg_loss_dis.append(loss_dis.data.item())
					loss_dis.backward()
					optimizer_dis.step()
				logger.info('epoch %s, dis-loss=%s' % (e, float(sum(avg_loss_dis))/len(avg_loss_dis)))
				for g in range(GEN_STEPS):
					generator.zero_grad()
					data_batch, label_batch  = dataloader.__next__()
					input_batch = Variable(data_batch).to(device)
					label_batch = Variable(label_batch).to(device)
					gen_output = generator(input_batch).detach()
					dis_input = Variable(gen_output, requires_grad=True).to(device)
					loss_adversarial = criterion_gen(dis_input, label_batch)
					input_fake_batch = torch.cat((input_batch, gen_output), 1)
					output_batch = discriminator(input_fake_batch)
					loss_content = criterion_dis(output_batch, Variable(torch.zeros(BATCH_SIZE, 1)).to(device))
					loss_total = ALPHA * loss_adversarial + loss_content
					avg_loss_gen.append(loss_total.data.item())
					loss_total.backward()
					optimizer_gen.step()
				logger.info('epoch %s, gen-loss=%s' % (e, float(sum(avg_loss_gen))/len(avg_loss_gen)))
		except StopIteration as stoperror:
			pass
		except Exception as error:
			logger.error(error)
			# traceback.print_exc()
		finally:
			if e % SAVE_STEP==0 and e!=0:
				torch.save(generator, os.path.join(PARAM_PATH, 'generator.%s.pkl' % (e)))
				torch.save(discriminator, os.path.join(PARAM_PATH, 'discriminator.%s.pkl' % (e)))

if __name__ == '__main__':
	# get_logger()
	train()