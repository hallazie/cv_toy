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

def test():
	data = tf.placeholder(dtype=tf.float32, name='data')
	label = tf.placeholder(dtype=tf.float32, name='label')
	trainable = tf.placeholder(dtype=tf.bool, name='trainable')
	thresh_gt = tf.placeholder(dtype=tf.float32, name='thresh_gt')
	thresh_ig = tf.placeholder(dtype=tf.float32, name='thresh_ig')

	palletnet = PalletNet(data, label, trainable, thresh_gt, thresh_ig)
	_, outsyb = palletnet.loss()
	variables = tf.global_variables()
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		saver.restore(sess, 'model/pallet_4999_loss=0.0015.ckpt-4999')
		output = sess.run([outsyb], feed_dict={
			data		:	dat[0],
			label		:	dat[1],
			trainable	:	True,
			thresh_ig	:	0.5,
			thresh_gt	:	0.9
		})		


if __name__ == '__main__':
	try:
		test()
	except Exception as e:
		traceback.print_exc()
