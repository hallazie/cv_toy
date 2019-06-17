# coding:utf-8
# author:xsh

import tensorflow as tf
import numpy as np
import cv2
import traceback

from tqdm import tqdm

from model import *
from dataiter import *
import config as cfg
import utils

def train():
	data = tf.placeholder(dtype=tf.float32, name='data')
	label = tf.placeholder(dtype=tf.float32, name='label')
	trainable = tf.placeholder(dtype=tf.bool, name='trainable')
	thresh_gt = tf.placeholder(dtype=tf.float32, name='thresh_gt')
	thresh_ig = tf.placeholder(dtype=tf.float32, name='thresh_ig')
	palletnet = PalletNet(data, label, trainable, thresh_gt, thresh_ig)
	symbol = palletnet.loss()

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	sess.run(tf.global_variables_initializer())
	diter = Dataiter()
	# saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
	
	trainable_list = []
	for var in tf.trainable_variables():
		var_name = var.op.name
		var_mess = str(var_name).split('/')
		trainable_list.append(var)
	optimizer = tf.train.AdamOptimizer(1e-3).minimize(symbol, var_list=trainable_list)

	for e in range(cfg.EPOCH):
		train_loss = []
		# pbar = tqdm(diter)
		# for dat in pbar:
		# print('EPOCH: %s' % e)
		for dat in diter:
			# print(dat.shape)
			print('average: %s' % np.mean(dat[0]))
			curr_loss = sess.run([symbol], feed_dict={
				data		:	dat[0],
				label		:	dat[1],
				trainable	:	True,
				thresh_ig	:	0.5,
				thresh_gt	:	0.9
			})
			train_loss.append(curr_loss)
			print('train loss: %s' % curr_loss)	
			# pbar.set_description('train loss: %.2f' %curr_loss)
			if (e+1)%10 == 0:
				ckpt_file = "params/pallet_%s_loss=%.4f.ckpt" % (e, curr_loss)	
				# saver.save(sess, ckpt_file, global_step=e)

if __name__ == '__main__':
	try:
		train()
	except Exception as e:
		traceback.print_exc()
