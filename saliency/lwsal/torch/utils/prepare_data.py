# coding : utf-8
# @author : xsh

import cv2
import os

raw_data_path = 'E:/data/saliency/SALICON/Train'
raw_label_path = 'E:/data/saliency/SALICON/TrainFix'
fine_data_path = 'E:/data/saliency/SALICON/Crop/fine'
coarse_data_path = 'E:/data/saliency/SALICON/Crop/coarse'
crop_label_path = 'E:/data/saliency/SALICON/Crop/label'


def gen_data():
	for _,_,fs in os.walk(raw_data_path):
		for idx, f in enumerate(fs):
			prefix = f.split('.')[0]
			data_path = os.path.join(raw_data_path, prefix + '.jpg')
			label_path = os.path.join(raw_label_path, prefix + '.jpeg')
			data_raw = cv2.imread(data_path)
			label_raw = cv2.imread(label_path)
			data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
			for i in range(6):
				for j in range(8):
					crop_fine = data_raw[i*80:(i+1)*80, j*80:(j+1)*80]
					crop_coarse = data_pad[i*80:(i+3)*80, j*80:(j+3)*80]
					crop_label = label_raw[i*80:(i+1)*80, j*80:(j+1)*80]
					cv2.imwrite(os.path.join(fine_data_path, prefix + '_' + str(i*8+j) + '.jpg'), crop_fine)
					cv2.imwrite(os.path.join(coarse_data_path, prefix + '_' + str(i*8+j) + '.jpg'), cv2.resize(crop_coarse, (80, 80), interpolation=cv2.INTER_LANCZOS4))
					cv2.imwrite(os.path.join(crop_label_path, prefix + '_' + str(i*8+j) + '.jpg'), crop_label)
			print('[INFO] %s/10000 finished' % idx)

if __name__ == '__main__':
	gen_data()