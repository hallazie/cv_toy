# coding:utf-8
# 
# @author: xsh

from torch.utils.data import Dataset

import numpy as np
import cv2
import os
import traceback

def img_loader(imgpath):
	return cv2.imread(imgpath)

class SaliconSet(Dataset):
	def __init__(self, data_path, label_path, img_transform=None, loader=img_loader):
		self.data_path = data_path
		self.label_path = label_path
		self.img_list = []
		for _,_,fs in os.walk(data_path):
			self.img_list = fs
			break
		self.img_transform = img_transform
		self.loader = loader
		self.jitter = 1e-10

	def __getitem__(self, idx):
		try:
			file_name = self.img_list[idx]
			curr_data_path = os.path.join(self.data_path, file_name)
			curr_label_path = os.path.join(self.label_path, file_name)

			# curr_data_batch = np.swapaxes(cv2.imread(curr_data_path), 0, 2).astype(np.float32)
			# curr_label_batch = np.expand_dims(cv2.resize(cv2.imread(curr_label_path), (40, 30), interpolation=cv2.INTER_LANCZOS4)[:,:,0].transpose(), axis=0).astype(np.float32)

			curr_data_batch = self.normalize(np.zeros((3, 640, 480)))
			curr_label_batch = self.normalize_label(np.zeros((1,40,30)))
			return curr_data_batch, curr_label_batch
		except Exception as e:
			traceback.print_exc()

	def normalize(self, arr):
		return (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + self.jitter)

	def normalize_label(self, arr):
		return arr / 255.

	def __len__(self):
		return len(self.img_list)