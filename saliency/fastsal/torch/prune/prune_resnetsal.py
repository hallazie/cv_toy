# --*-- coding:utf-8 --*--
# 
# @author:xsh

import sys
sys.path.append('..')


import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet50, Bottleneck
from thop import profile


CUDA = torch.cuda.is_available()
PRUNE_SAVE_PATH = '.'
CHECKPOINT_PATH = 'E:\\Dataset\\SAL\\output\\resnetsal'

class _ScaleUp(nn.Module):

	def __init__(self, in_size, out_size):
		super(_ScaleUp, self).__init__()

		self.up = nn.Sequential(
			nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2),
			nn.BatchNorm2d(in_size),
			nn.LeakyReLU(inplace=True))
		self.conv = nn.Sequential(
			nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=2, dilation=2),
			nn.BatchNorm2d(out_size),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(out_size, out_size, kernel_size=1, stride=1),
			nn.LeakyReLU(inplace=True),
		)

		self.__init_weights__()

	def __init_weights__(self):

		nn.init.kaiming_normal_(self.conv[0].weight)
		nn.init.constant_(self.conv[0].bias, 0.0)
		nn.init.kaiming_normal_(self.conv[3].weight)
		nn.init.constant_(self.conv[3].bias, 0.0)
		nn.init.kaiming_normal_(self.up[0].weight)
		nn.init.constant_(self.up[0].bias, 0.0)

	def forward(self, inputs):
		output = self.up(inputs)
		output = self.conv(output)
		return output


class ResNetSal(nn.Module):

	def __init__(self, ):
		super(ResNetSal, self).__init__()
		self.encode_image = resnet50(pretrained=False)
		modules = list(self.encode_image.children())[:-2]
		modules_flat = []
		for m in modules:
			if type(m) == nn.Sequential:
				for x in list(m.children()):
					modules_flat.append(x)
			else:
				modules_flat.append(m)
		for m in modules_flat:
			if type(m) != Bottleneck:
				continue
			print(m)
			print('---------------')
		self.encode_image = nn.Sequential(*modules)
		self.decoder1 = _ScaleUp(2048, 1024)
		self.decoder2 = _ScaleUp(1024, 512)
		self.decoder3 = _ScaleUp(512, 256)
		self.saliency = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

		self.__init_weights__()

	def __init_weights__(self):

		nn.init.kaiming_normal_(self.saliency.weight)
		nn.init.constant_(self.saliency.bias, 0.0)

	def forward(self, x):
		x1 = self.encode_image(x)
		x1 = self.decoder1(x1)
		x1 = self.decoder2(x1)
		x1 = self.decoder3(x1)
		sal = self.saliency(x1)
		sal = F.relu(sal, inplace=True)
		return sal

if __name__ == '__main__':

	model = ResNetSal()

	# if CUDA:
	# 	model.cuda()

	# if os.path.isfile(CHECKPOINT_PATH):
	# 	print("=> loading checkpoint '{}'".format(CHECKPOINT_PATH))
	# 	checkpoint = torch.load(CHECKPOINT_PATH)
	# 	best_prec1 = checkpoint['best_prec1']
	# 	model.load_state_dict(checkpoint['state_dict'])
	# 	print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(CHECKPOINT_PATH, checkpoint['epoch'], best_prec1))
	# else:
	# 	print("=> no checkpoint found at '{}'".format(CHECKPOINT_PATH))

	# print('Pre-processing Successful!')

	# inp = torch.randn(1, 3, 320, 256)
	# flops, params = profile(model.cpu(), inputs=((inp), ))
	# g_flops = flops / float(1024*1024*1024)
	# m_params = params / float(1024*1024)
	# line_0 = 'raw [%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnet', round(g_flops, 4), round(m_params, 4))
   

	# skip = {'A': []}

	# prune_prob = {'A': 0.75}

	# layer_id = 1
	# cfg = []
	# cfg_mask = []
	# for m in model.modules():
	# 	if isinstance(m, nn.Conv2d):
	# 		out_channels = m.weight.data.shape[0]
	# 		if layer_id in skip[args.v]:
	# 			cfg_mask.append(torch.ones(out_channels))
	# 			cfg.append(out_channels)
	# 			layer_id += 1
	# 			continue
	# 		if layer_id % 2 == 0:
	# 			stage = layer_id // 36
	# 			if layer_id <= 36:
	# 				stage = 0
	# 			elif layer_id <= 72:
	# 				stage = 1
	# 			elif layer_id <= 108:
	# 				stage = 2
	# 			prune_prob_stage = prune_prob[args.v][stage]
	# 			weight_copy = m.weight.data.abs().clone().cpu().numpy()
	# 			L1_norm = np.sum(weight_copy, axis=(1,2,3))
	# 			num_keep = int(out_channels * (1 - prune_prob_stage))
	# 			arg_max = np.argsort(L1_norm)
	# 			arg_max_rev = arg_max[::-1][:num_keep]
	# 			mask = torch.zeros(out_channels)
	# 			mask[arg_max_rev.tolist()] = 1
	# 			cfg_mask.append(mask)
	# 			cfg.append(num_keep)
	# 			layer_id += 1
	# 			continue
	# 		layer_id += 1

	# newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
	# if args.cuda:
	# 	newmodel.cuda()

	# start_mask = torch.ones(3)
	# layer_id_in_cfg = 0
	# conv_count = 1
	# for [m0, m1] in zip(model.modules(), newmodel.modules()):
	# 	if isinstance(m0, nn.Conv2d):
	# 		if conv_count == 1:
	# 			m1.weight.data = m0.weight.data.clone()
	# 			conv_count += 1
	# 			continue
	# 		if conv_count % 2 == 0:
	# 			mask = cfg_mask[layer_id_in_cfg]
	# 			idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
	# 			if idx.size == 1:
	# 				idx = np.resize(idx, (1,))
	# 			w = m0.weight.data[idx.tolist(), :, :, :].clone()
	# 			m1.weight.data = w.clone()
	# 			layer_id_in_cfg += 1
	# 			conv_count += 1
	# 			continue
	# 		if conv_count % 2 == 1:
	# 			mask = cfg_mask[layer_id_in_cfg-1]
	# 			idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
	# 			if idx.size == 1:
	# 				idx = np.resize(idx, (1,))
	# 			w = m0.weight.data[:, idx.tolist(), :, :].clone()
	# 			m1.weight.data = w.clone()
	# 			conv_count += 1
	# 			continue
	# 	elif isinstance(m0, nn.BatchNorm2d):
	# 		if conv_count % 2 == 1:
	# 			mask = cfg_mask[layer_id_in_cfg-1]
	# 			idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
	# 			if idx.size == 1:
	# 				idx = np.resize(idx, (1,))
	# 			m1.weight.data = m0.weight.data[idx.tolist()].clone()
	# 			m1.bias.data = m0.bias.data[idx.tolist()].clone()
	# 			m1.running_mean = m0.running_mean[idx.tolist()].clone()
	# 			m1.running_var = m0.running_var[idx.tolist()].clone()
	# 			continue
	# 		m1.weight.data = m0.weight.data.clone()
	# 		m1.bias.data = m0.bias.data.clone()
	# 		m1.running_mean = m0.running_mean.clone()
	# 		m1.running_var = m0.running_var.clone()
	# 	elif isinstance(m0, nn.Linear):
	# 		m1.weight.data = m0.weight.data.clone()
	# 		m1.bias.data = m0.bias.data.clone()

	# torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

	# print(newmodel)
	# model = newmodel
	# acc = test(model.cuda())

	# num_parameters = sum([param.nelement() for param in newmodel.parameters()])
	# print("number of parameters: "+str(num_parameters))
	# with open(os.path.join(args.save, "prune.txt"), "w") as fp:
	# 	fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
	# 	fp.write("Test Accuracy: \n"+str(acc)+"\n")


	# inp = torch.randn(1,3,32,32)
	# flops, params = profile(model.cpu(), inputs=((inp), ))
	# g_flops = flops / float(1024*1024*1024)
	# m_params = params / float(1024*1024)
	# line_1 = 'proned[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnet-prune', round(g_flops, 4), round(m_params, 4))

	# print('\n----------------------')
	# print(line_0)
	# print(line_1)
	# print('----------------------\n')