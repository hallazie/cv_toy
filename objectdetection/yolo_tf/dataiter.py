# coding:utf-8
# @author:xsh

import os
import random
import cv2
import traceback
import json
import numpy as np
import tensorflow as tf

import config as cfg
import utils

class Dataiter(object):
	def __init__(self):
		self.anchor = utils.get_anchor(cfg.LABELPATH)
		self.file_list = []
		self.json_list = []
		self.get_list()
		self.batch_done = 0
		self.batch_num = len(self.file_list) // cfg.BATCHSIZE
		print('total: %s, bsize: %s, iters: %s' % (len(self.file_list), cfg.BATCHSIZE, self.batch_num))

	def __iter__(self):
		return self

	def __next__(self):
		# with tf.device('/cpu:0'):
		try:
			batch_image = np.zeros((cfg.BATCHSIZE, 1, cfg.INPUTHEIGHT, cfg.INPUTWIDTH))
			batch_label = np.zeros((cfg.BATCHSIZE, 5, cfg.INPUTHEIGHT//cfg.DOWNSCALE, cfg.INPUTWIDTH//cfg.DOWNSCALE))
			if self.batch_done < self.batch_num:
				cnt = 0
				while cnt < cfg.BATCHSIZE:
					prefix = self.file_list[self.batch_done * cfg.BATCHSIZE + cnt]
					curr_img = utils.normalize_img(cv2.resize(cv2.imread(os.path.join(cfg.DATAPATH, prefix + '.png'), cv2.IMREAD_GRAYSCALE), (cfg.INPUTWIDTH, cfg.INPUTHEIGHT), interpolation=cv2.INTER_LANCZOS4))
					batch_image[cnt] = curr_img
					if prefix in self.json_list:
						curr_jsn = json.load(open(os.path.join(cfg.DATAPATH, prefix + '.json')))
						curr_lbl = utils.jbox_2_label((cfg.INPUTWIDTH//cfg.DOWNSCALE, cfg.INPUTHEIGHT//cfg.DOWNSCALE), curr_jsn, self.anchor)
						batch_label[cnt] = curr_lbl.transpose()
					else:
						batch_label[cnt] = np.zeros((5, cfg.INPUTHEIGHT//cfg.DOWNSCALE, cfg.INPUTWIDTH//cfg.DOWNSCALE))
					cnt += 1
				return batch_image, batch_label
			else:
				self.batch_done = 0
				random.shuffle(self.file_list)
				raise StopIteration
		except:
			traceback.print_exc()

	def next(self):
		return self.__next__()

	def get_list(self):
		file_set = set()
		for _,_,fs in os.walk(cfg.DATAPATH):
			for f in fs:
				prefix = f.split('.')[0]
				file_set.add(prefix)
				if f.endswith('json'):
					self.json_list.append(prefix)
			self.file_list = list(file_set)
			random.shuffle(self.file_list)
			break
