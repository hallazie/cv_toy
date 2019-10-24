# coding:utf-8
#
# @author:xsh
#
# Generating gaussian blur images

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter

def gen():
	# size: 256x256
	# 6,6	6,12	6,18	6,24
	# 12,6	12,12	12,18	12,24
	# 18,6	18,12	18,18	18,24
	# 24,6	24,12	24,18	24,24
	for i in range(1,5):
		for j in range(1,5):
			img = np.zeros((256, 256)).astype(np.float32)
			img[127, 127] = 255.
			img[127, 128] = 255.
			img[128, 127] = 255.
			img[128, 128] = 255.
			blur = gaussian_filter(img, sigma=(i*6, j*6))
			blur = 255. * (blur - np.min(blur)) / (np.max(blur) - np.min(blur))
			cv2.imwrite('%s,%s.jpg' % (i, j), blur)

if __name__ == '__main__':
	gen()