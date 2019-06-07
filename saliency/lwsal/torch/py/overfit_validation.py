# coding:utf-8
# @author: xsh

import torch
import torch.optim as optim

import logging

from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image as im

from model import *
from dataiter import *

logger_name = 'train_logger'
logger = logging.getLogger(logger_name)

FINE_PATH = 'E:/data/saliency/SALICON/Overfit/fine'
COARSE_PATH = 'E:/data/saliency/SALICON/Overfit/coarse'
LABEL_PATH = 'E:/data/saliency/SALICON/Overfit/label'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/params'
OUTPUT_PATH = 'E:/cv_toy/saliency/lwsal/output'
PLACEHOLDER = 'E:/cv_toy/saliency/lwsal/data/placeholder.jpg'
BATCH_SIZE = 48
EPOCHES = 1000
GRAD_ACCUM = 1
JITTER = 1e-7

def normalize_255(arr):
	if np.min(arr == np.max(arr)):
		return np.clip(255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + JITTER), 0, 255)
	else:
		return np.clip(255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr)), 0, 255)

def normalize(arr):
	if np.min(arr == np.max(arr)):
		return np.clip((arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + JITTER), 0, 1)
	else:
		return np.clip((arr - np.min(arr)) / float(np.max(arr) - np.min(arr)), 0, 1)

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
			if True:
				optimizer.step()
				optimizer.zero_grad()
		if (e + 1) % 100 == 0:
			torch.save(model.state_dict(), os.path.join(PARAM_PATH, 'model_%s.pkl' % str(e + 1)))

def inference():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Net().to(device)
	model.load_state_dict(torch.load(os.path.join(PARAM_PATH, 'model_%s.pkl' % EPOCHES)))
	model.eval()
	placeholder = cv2.resize(cv2.imread(PLACEHOLDER), (10, 10))[:,:,2].reshape((1,1,10,10)).astype(np.float32)

	f = 'COCO_train2014_000000000110.jpg'
	prefix = f.split('.')[0]
	data_path = os.path.join('E:/data/saliency/SALICON/Overfit/', prefix + '.jpg')
	data_raw = cv2.imread(data_path)
	data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
	res = np.zeros((480//8, 640//8))
	res_fine = np.zeros((480//8, 640//8))
	res_coarse = np.zeros((480//8, 640//8))
	for i in range(6):
		for j in range(8):
			crop_fine = np.expand_dims(np.swapaxes(data_raw[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2), axis=0).astype(np.float32)
			crop_coarse = np.expand_dims(np.swapaxes(cv2.resize(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], (80, 80), interpolation=cv2.INTER_LANCZOS4), 0, 2), axis=0).astype(np.float32)
			fine = Variable(torch.FloatTensor(normalize(crop_fine)).to(device))
			coarse = Variable(torch.FloatTensor(normalize(crop_coarse)).to(device))
			label = Variable(torch.FloatTensor(normalize(placeholder)).to(device), requires_grad=False)
			_, output, output_fine, output_coarse = model(fine, coarse, label)
			ret = output.detach().cpu().numpy()[0][0].transpose()
			res[i*10:(i+1)*10, j*10:(j+1)*10] = ret
			res_fine[i*10:(i+1)*10, j*10:(j+1)*10] = output_fine.detach().cpu().numpy()[0][0].transpose()
			res_coarse[i*10:(i+1)*10, j*10:(j+1)*10] = output_coarse.detach().cpu().numpy()[0][0].transpose()
			# cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_%s.jpg' % str(i*8+j)), cv2.resize(ret, (160, 160)))
	print('[DEBUG] fine range: %s~%s, coarse range: %s~%s' % (np.min(res_fine), np.max(res_fine), np.min(res_coarse), np.max(res_coarse)))
	# res = cv2.blur(res, (5,5))
	res = normalize_255(res)
	res_fine = normalize_255(res_fine)
	res_coarse = normalize_255(res_coarse)
	# res = cv2.resize(normalize_255(res), (640, 480), interpolation=cv2.INTER_LANCZOS4)
	# res = cv2.blur(res, (25,25))
	cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_p.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
	cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_f.jpg'), cv2.resize(res_fine.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
	cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_c.jpg'), cv2.resize(res_coarse.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
	im.fromarray(res.astype(np.uint8)).resize((640, 480)).save(os.path.join(OUTPUT_PATH, prefix + '_p_im.jpg'))
	cv2.imwrite(os.path.join(OUTPUT_PATH, f), data_raw)
	print('[DEBUG] %s inference finished' % f)

if __name__ == '__main__':
	get_logger()
	train()
	inference()