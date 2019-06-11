# coding:utf-8
# author:xsh

import tensorflow as tf
import numpy as np

import util

class PalletNet(Object):
	def __init__(self, data, train):
		self.data = data
		self.train = train
		self.anchors = util.get_anchor()

	def backbone(self, data):
		pass

	def conv_block(self, num_filter, kernel, stride, pad):
		pass

	def pool_block(self, kernel, stride):
		pass

	def __build_network(self, data):
		pass

	def decode_out_tensor(self):
		pass

	def loss(self):
		pass