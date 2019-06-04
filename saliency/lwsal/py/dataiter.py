# coding:utf-8
# @author: xsh

from torch.utils.data import Dataset
from PIL import Image as im

import os

def img_loader(imgpath):
	return im.open(imgpath)

def SaliconSet(Dataset):
	def __init__(self, fine_path, coarse_path, label_path, img_transform=None, loader=img_loader):
		self.fine_path = fine_path
		self.coarse_path = coarse_path
		self.label_path = label_path
		self.img_list = []
		for _,_,fs in os.walk(fine_path):
			self.img_list = fs.copy()
			break
		self.img_transform = img_transform
		self.loader = loader

	def __get_item__(self, idx):
		file_name = self.img_list[idx]
		curr_fine_path = os.path.join(self.fine_path, file_name)
		curr_coarse_path = os.path.join(self.coarse_path, file_name)
		curr_label_path = os.path.join(self.label_path, file_name)
		return curr_fine_path, curr_coarse_path, curr_label_path

	def __len__(self):
		return len(self.img_list)
