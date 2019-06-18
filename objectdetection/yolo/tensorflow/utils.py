import json
import os
import numpy as np
import math

import config as cfg

def normalize_img(arr):
	return (arr - np.min(arr)) / float(np.max(arr) - np.min(arr))

def normalize_255(arr):
	if np.min(arr == np.max(arr)):
		return np.clip(255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + 1e-7), 0, 255)
	else:
		return np.clip(255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr)), 0, 255)

def get_anchor(json_path):
	coords = []
	for _,_,fs in os.walk(json_path):
		for f in fs:
			if not f.endswith('.json'):
				continue
			with open(json_path+f, 'r') as jfile:
				dat = json.load(jfile)
				w = abs(dat['shapes'][0]['points'][0][0] - dat['shapes'][0]['points'][1][0])
				h = abs(dat['shapes'][0]['points'][0][1] - dat['shapes'][0]['points'][1][1])
				coords.append((w,h))
	return (sum([e[0] for e in coords])/len(coords), sum([e[1] for e in coords])/len(coords))

def jbox_2_label(grid_shape, jbox, anchor):
	label = np.zeros((grid_shape[0], grid_shape[1], 5))
	cnt = 0
	for i in range(len(jbox['shapes'])):
		x0, y0 = jbox['shapes'][i]['points'][0][0], jbox['shapes'][i]['points'][0][1]
		x1, y1 = jbox['shapes'][i]['points'][1][0], jbox['shapes'][i]['points'][1][1]
		xa, ya = float(x0+x1)/2., float(y0+y1)/2.
		w0, h0 = abs(x0-x1), abs(y0-y1)
		xc, yc = int(xa//cfg.DOWNSCALE)*cfg.DOWNSCALE, int(ya//cfg.DOWNSCALE)*cfg.DOWNSCALE
		bx, by = (xa-xc)/float(cfg.DOWNSCALE), (ya-yc)/float(cfg.DOWNSCALE)
		bw, bh = math.log(w0/float(anchor[0])), math.log(h0/float(anchor[1]))
		cur_row = [bx, by, bw, bh, 1]
		label[int(xa//cfg.DOWNSCALE), int(ya//cfg.DOWNSCALE)] = np.array(cur_row)
		cnt += 1
	return label
