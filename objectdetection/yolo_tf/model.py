# coding:utf-8
# author:xsh

import tensorflow as tf
import numpy as np

# import util

class PalletNet(object):
	def __init__(self, data, train):
		self.data = data
		self.train = train
		self.anchors = util.get_anchor()

		self.cv_box = self.build_network(self.data)
		self.bd_box = self.decode(self.cv_box, self.anchors)
		

	def backbone(self, tensor):
		tensor = self.conv_block(True, tensor, (3, 3, 3, 32))
		tensor = self.pool_block(tensor)
		tensor = self.conv_block(True, tensor, (32, 3, 3, 64))
		tensor = self.pool_block(tensor)
		tensor = self.conv_block(True, tensor, (64, 3, 3, 96))
		tensor = self.pool_block(tensor)
		tensor = self.conv_block(True, tensor, (96, 3, 3, 128))
		tensor = self.pool_block(tensor)
		tensor = self.conv_block(True, tensor, (128, 3, 3, 128))
		tensor = self.conv_block(True, tensor, (128, 1, 1, 5*3))

	def upsp_block(self, data):
		num_filter = data.shape.as_list()[-1]
		tensor = tf.layers.conv2d_transpose(
			data,
			num_filter,
			kernel_size=2,
			padding='same',
			strides=(2,2),
			kernel_initializer=tf.random_normal_initializer()
		)
		return tensor

	def decv_block(self, trainable, data, filter_shapes, output_shape, strides):
		weight = tf.get_variable(
			name='weight',
			dtype=tf.float32,
			trainable=True,
			shape=filter_shapes,
			initializer=tf.random_normal_initializer(stddev=0.01)
		)
		tensor = tf.nn.conv2d_transpose(
			input=data,
			filter=weight,
			output_shape=output_shape,
			strides=strides,
			padding='SAME'
		)
		tensor = tf.layers.batch_normalization(
			tensor,
			beta_initializer=tf.zero_initializer(),
			gamma_initializer=tf.one_initializer(),
			moving_mean_initializer=tf.zero_initializer(),
			moving_variance_initializer=tf.one_initializer(),
			trainable=trainable
		)
		tensor = tf.nn.leaky_relu(tensor, alpha=0.1)
		return tensor

	def conv_block(self, trainable, data, filter_shapes, strides=(1,1,1,1), padding='SAME'):
		weight = tf.get_variable(
			name='weight',
			dtype=tf.float32,
			trainable=True,
			shape=filter_shapes,
			initializer=tf.random_normal_initializer(stddev=0.01)
		)
		tensor = tf.nn.conv2d(
			input=data,
			filter=weight,
			strides=strides,
			padding=padding
		)
		tensor = tf.layers.batch_normalization(
			tensor,
			beta_initializer=tf.zero_initializer(),
			gamma_initializer=tf.one_initializer(),
			moving_mean_initializer=tf.zero_initializer(),
			moving_variance_initializer=tf.one_initializer(),
			trainable=trainable
		)
		tensor = tf.nn.leaky_relu(tensor, alpha=0.1)
		return tensor

	def pool_block(self, data):
		return tf.nn.max_pool(data, (1,2,2,1), (1,2,2,1), 'VALID')

	def build_network(self, data):
		pass

	def decode(self, data, anchors):
		conv_shape = tf.shape(data)
		batch_size = conv_shape[0]
		output_height = conv_shape[1]
		output_width = conv_shape[2]
		anchor_nums = len(anchors)

		data = tf.reshape(data, (batch_size, output_height, output_width, anchor_nums, 5))
		data_raw_dxdy = data[:, :, :, :, 0:2]
		data_raw_dwdh = data[:, :, :, :, 2:4]
		data_raw_conf = data[:, :, :, :, 5:]

		x = tf.tile(tf.range(output_height, dtype=tf.int32)[:, tf.newaxis], [1, output_width])
		y = tf.tile(tf.range(output_width, dtype=tf.int32)[tf.newaxis, :], [output_height, 1])
		xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
		xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_nums, 1])
		xy_grid = tf.cast(xy_grid, tf.float32)

		pred_xy = (tf.sigmoid(data_raw_dxdy) + xy_grid)
		pred_wh = (tf.exp(data_raw_dwdh) * anchors)
		pred_cf = tf.sigmoid(data_raw_conf)
		return tf.concat([pred_xy, pred_wh, pred_cf], axis=-1)

	def loss(self):
		pass