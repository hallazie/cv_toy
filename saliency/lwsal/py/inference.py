# coding:utf-8
# @author: xsh

import torch
import torch.optim as optim

import os
import cv2
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *

VALID_PATH = 'E:/data/saliency/SALICON/Valid'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/params'
OUTPUT_PATH = 'E:/cv_toy/saliency/lwsal/output'
PLACEHOLDER = 'E:/cv_toy/saliency/lwsal/data/placeholder.jpg'
OVERFIT_PATH = 'E:/data/saliency/SALICON/Overfit'
EPOCH = 500
BATCH_SIZE = 128
EPOCHES = 1000
GRAD_ACCUM = 2

def magik(inp):
	ret = inp[1:9, 1:9]
	return cv2.copyMakeBorder(ret, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

def normalize_255(arr):
	return 255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr))

def normalize(arr):
	return (arr - np.min(arr)) / float(np.max(arr) - np.min(arr))

def inference():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# model = Net().to(device)
	model = torch.load(os.path.join(PARAM_PATH, 'model_%s.pkl' % EPOCH)).to(device)
	model.eval()
	placeholder = cv2.resize(cv2.imread(PLACEHOLDER), (10, 10))[:,:,2].reshape((1,1,10,10)).astype(np.float32)
	for _,_,fs in os.walk(VALID_PATH):
		for f in fs[:1]:
			# f = 'COCO_train2014_000000000110.jpg'
			prefix = f.split('.')[0]
			data_path = os.path.join(VALID_PATH, prefix + '.jpg')
			data_raw = cv2.imread(data_path)
			data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
			res = np.zeros((480//8, 640//8))
			for i in range(6):
				for j in range(8):
					crop_fine = np.expand_dims(np.swapaxes(data_raw[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2), axis=0).astype(np.float32)
					crop_coarse = np.expand_dims(np.swapaxes(cv2.resize(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], (80, 80), interpolation=cv2.INTER_LANCZOS4), 0, 2), axis=0).astype(np.float32)
					fine = Variable(torch.FloatTensor(normalize(crop_fine)).to(device))
					coarse = Variable(torch.FloatTensor(normalize(crop_coarse)).to(device))
					label = Variable(torch.FloatTensor(normalize(placeholder)).to(device), requires_grad=False)
					_, output, _, _ = model(fine, coarse, label)
					ret = output.detach().cpu().numpy()[0][0].transpose()
					res[i*10:(i+1)*10, j*10:(j+1)*10] = ret
					# cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_%s.jpg' % str(i*8+j)), cv2.resize(ret, (160, 160)))
			res = cv2.resize(normalize_255(res), (640, 480), interpolation=cv2.INTER_LANCZOS4)
			cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_p.jpg'), res)
			cv2.imwrite(os.path.join(OUTPUT_PATH, f), data_raw)
			print('[DEBUG] %s inference finished' % f)
			

if __name__ == '__main__':
	inference()