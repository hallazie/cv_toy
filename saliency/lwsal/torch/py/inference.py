# coding:utf-8
# @author: xsh

import torch
import torch.optim as optim

import os
import random
import cv2
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataiter import *

VALID_PATH = 'E:/data/saliency/SALICON/Valid'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/torch/params'
OUTPUT_PATH = 'E:/cv_toy/saliency/lwsal/torch/output'
PLACEHOLDER = 'E:/cv_toy/saliency/lwsal/torch/data/placeholder.jpg'
OVERFIT_PATH = 'E:/data/saliency/SALICON/Overfit'
EPOCH = 1
BATCH_SIZE = 128
EPOCHES = 1000
GRAD_ACCUM = 2
JITTER = 1e-7

def magik(inp):
	ret = inp[1:9, 1:9]
	return cv2.copyMakeBorder(ret, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

def normalize_255(arr):
	return 255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + JITTER)

def normalize(arr):
	return (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + JITTER)

def inference():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# model = Net().to(device)
	model = torch.load(os.path.join(PARAM_PATH, 'model_%s.pkl' % EPOCH)).to(device)
	model.eval()
	placeholder = cv2.resize(cv2.imread(PLACEHOLDER), (10, 10))[:,:,2].reshape((1,1,10,10)).astype(np.float32)
	for _,_,fs in os.walk(VALID_PATH):
		# random.shuffle(fs)
		for f in fs[:10]:
			# f = 'COCO_val2014_000000162952.jpg'
			prefix = f.split('.')[0]
			data_path = os.path.join(VALID_PATH, prefix + '.jpg')
			data_raw = cv2.imread(data_path)
			data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
			res = np.zeros((480//8, 640//8))
			res_fine = np.zeros((480//8, 640//8))
			res_coarse = np.zeros((480//8, 640//8))
			for i in range(6):
				for j in range(8):
					crop_fine = np.expand_dims(np.swapaxes(data_raw[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2), axis=0).astype(np.float32)
					crop_coarse = np.expand_dims(np.swapaxes(cv2.resize(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], (80, 80), interpolation=cv2.INTER_LANCZOS4), 0, 2), axis=0).astype(np.float32)
					fine = Variable(torch.FloatTensor(normalize(crop_fine)).to(device), requires_grad=False)
					coarse = Variable(torch.FloatTensor(normalize(crop_coarse)).to(device), requires_grad=False)
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
			# res = cv2.blur(res, (25,25))
			cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_p.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_f.jpg'), cv2.resize(res_fine.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			cv2.imwrite(os.path.join(OUTPUT_PATH, prefix + '_c.jpg'), cv2.resize(res_coarse.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			cv2.imwrite(os.path.join(OUTPUT_PATH, f), data_raw)
			print('[DEBUG] %s inference finished' % f)
			

if __name__ == '__main__':
	inference()