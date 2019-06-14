# coding:utf-8
# author:xsh

import tensorflow as tf
import numpy as np

import cv2

from tqdm import tqdm
from model import *
from dataiter import *

def train():
	data = tf.placeholder(dtype=tf.float32, name='data')
	label = tf.placeholder(dtype=tf.float32, name='label')
	trainable = tf.placeholder(dtype=tf.bool, name='trainable')
	thresh_gt = tf.placeholder(dtype=tf.float32, name='thresh_gt')
	thresh_ig = tf.placeholder(dtype=tf.float32, name='thresh_ig')
	palletnet = PalletNet(data, label, trainable, thresh_gt, thresh_ig)

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	sess.run(tf.global_variables_initializer())
	diter = Dataiter()
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

	for e in epoch:
		train_loss = []
		pbar = tqdm(diter)
		for dat in pbar:
			curr_loss = sess.run([palletnet], feed_dict={
				data		:	dat[0],
				label		:	dat[1],
				trainable	:	True,
				thresh_ig	:	0.5,
				thresh_gt	:	0.9
			})
			train_loss.append(curr_loss)
			pbar.set_description('train loss: %.2f' %curr_loss)
		if (e+1)%10 == 0:
			ckpt_file = "params/pallet_%s_loss=%.4f.ckpt" % (e, curr_loss)	
			saver.save(sess, ckpt_file, global_step=e)