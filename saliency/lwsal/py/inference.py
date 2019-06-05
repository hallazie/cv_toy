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
EPOCH = 10
BATCH_SIZE = 128
EPOCHES = 1000
GRAD_ACCUM = 2

def inference():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Net().to(device)
	model.load_state_dict(torch.load(os.path.join(PARAM_PATH, 'model_%s.pkl' % EPOCH)).to(device))
	placeholder = np.expand_dims(cv2.resize(cv2.imread(PLACEHOLDER), (10, 10))[:,:2], axis=0).astype(np.float32)
	for _,_,fs in os.walk(VALID_PATH):
		for f in fs[:10]:
			prefix = f.split('.')[0]
			data_path = os.path.join(VALID_PATH, prefix + '.jpg')
			data_raw = cv2.imread(data_path)
			data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
			res = np.zeros((640, 480))
			for i in range(6):
				for j in range(8):
					crop_fine = np.swapaxes(data_raw[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2).astype(np.float32)
					crop_coarse = np.swapaxes(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], 0, 2).astype(np.float32)
					fine = Variable(data_raw.to(device))
					coarse = Variable(data_pad.to(device))
					label = Variable(placeholder.to(device), requires_grad=False)
					_, output = model(fine, coarse, label)
					res[i*80:(i+1)*80, j*80:(j+1)*80] = output
			cv2.imwrite(os.path.join(OUTPUT_PATH, f), res)
			print('[DEBUG] %s inference finished' % f)

if __name__ == '__main__':
	inference()