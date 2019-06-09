import mxnet as mx
import numpy as np

import cv2
import os
import random
import logging

from model import *
from dataiter import *

FINE_PATH = 'E:/data/saliency/SALICON/Crop/fine'
COARSE_PATH = 'E:/data/saliency/SALICON/Crop/coarse'
LABEL_PATH = 'E:/data/saliency/SALICON/Crop/label'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/mxnet/params'
MODEL_PREFIX = 'params/lwsp'
BATCH_SIZE = 128
EPOCHES = 500
CTX = mx.gpu(0)

logging.getLogger().setLevel(logging.DEBUG)

def train():
	diter = SaliconIter(FINE_PATH, COARSE_PATH, LABEL_PATH, BATCH_SIZE)
	symbol = net()
	arg_names = symbol.list_arguments()
	arg_shapes, output_shapes, _ = symbol.infer_shape(fine=(1, 3, 80, 80), coarse=(1, 3, 80, 80))
	# for name, shape in zip(arg_names, arg_shapes):
	# 	print('%s\t\t%s' % (name, str(shape)))
	# print('out\t\t%s' % str(output_shapes))
	model = mx.mod.Module(symbol=symbol, context=CTX, data_names=('fine','coarse'), label_names=('label',))
	model.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)
	model.init_params(initializer=mx.init.Uniform(scale=.1))
	model.fit(
		diter,
		optimizer = 'adam',
		optimizer_params = {'learning_rate':1e-3},
		eval_metric = 'mse',
		batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 1),
		epoch_end_callback = mx.callback.do_checkpoint(MODEL_PREFIX, 1),
		num_epoch = EPOCHES,
	)


if __name__ == '__main__':
	train()