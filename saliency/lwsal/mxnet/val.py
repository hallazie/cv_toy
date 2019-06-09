# def it(n):
# 	for i in range(n):
# 		yield i**2

# iterator = it(10)
# for i in range(20):
# 	try:
# 		p = next(iterator)
# 		print(p)
# 		print(type(p))
# 	except StopIteration:
# 		print('just stoped')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def avg_hist():
	alst = []
	for _,_,fs in os.walk('E:/data/saliency/SALICON/ValidFix'):
		sums, counts = 0, 0
		for f in fs:
			p = cv2.imread(os.path.join('E:/data/saliency/SALICON/ValidFix', f))
			a = np.mean(p)
			alst.append(a)
			sums += a
			counts += 1
			print('current: %s' % a)
		print('average: %s' % (sums/counts))
	plt.hist(x=alst, bins=50)
	plt.show()

if __name__ == '__main__':
	avg_hist()