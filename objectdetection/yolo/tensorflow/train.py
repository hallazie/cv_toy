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
	variables = tf.global_variables()

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	# sess.run(tf.global_variables_initializer())
	diter = Dataiter()
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
	
	trainable_list = []
	for var in tf.trainable_variables():
		var_name = var.op.name
		var_mess = str(var_name).split('/')
		trainable_list.append(var)
	# for var in tf.global_variables():
	for var in tf.trainable_variables():
		print('var:\t%s' % str(var))
	optimizer = tf.train.AdamOptimizer(1e-3)
	train_step = optimizer.minimize(symbol, var_list=trainable_list)
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		with tf.control_dependencies([train_step]):
			train_op = tf.no_op()
	
	sess.run(tf.global_variables_initializer())

	for e in range(cfg.EPOCH):
		train_loss = []
		for b, dat in enumerate(diter):
			curr_loss = sess.run([train_step, symbol], feed_dict={
				data		:	dat[0],
				label		:	dat[1],
				trainable	:	True,
				thresh_ig	:	0.5,
				thresh_gt	:	0.9
			})
			train_loss.append(curr_loss)
			print('train loss at epoch\t%s, batch\t%s: %s' % (e, b, str(curr_loss[1])))
			# pbar.set_description('train loss: %.2f' %curr_loss)
		if (e+1)%10 == 0:
			ckpt_file = "params/pallet_%s_loss=%.4f.ckpt" % (e, curr_loss[1])	
			saver.save(sess, ckpt_file, global_step=e)

if __name__ == '__main__':
	try:
		train()
	except Exception as e:
		traceback.print_exc()
