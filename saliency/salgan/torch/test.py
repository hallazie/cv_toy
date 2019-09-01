# coding:utf-8
# 
# @author:xsh

import torch
import cv2
import numpy as np
import os

from config import *

def normalize(arr):
	return (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + 1e-7)

def normalize_255(arr):
	return 255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + 1e-7)

def test(e):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	generator = torch.load(os.path.join(PARAM_PATH, 'generator.%s.pkl' % e)).to(device)
	for _,_,fs in os.walk(DATA_PATH):
		for file_name in fs:
			if len(file_name) < 8:
				continue
			data_path = os.path.join(DATA_PATH, file_name)
			data_batch = np.swapaxes(cv2.resize(cv2.imread(data_path), (256, 192), interpolation=cv2.INTER_LANCZOS4), 0, 2).astype(np.float32)
			data_batch = np.expand_dims(normalize(data_batch), axis=0)
			output = generator(torch.tensor(data_batch).to(device)).detach()
			salmap = normalize_255(np.array(output[0][0].cpu()))
			cv2.imwrite(OUTPUT_PATH + file_name, cv2.resize(salmap.transpose(), (640, 480), interpolation=cv2.INTER_LANCZOS4))

if __name__ == '__main__':
	test(100)