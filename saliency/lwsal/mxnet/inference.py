import mxnet
import cv2
import random

from model import *
from dataiter import *

FINE_PATH = 'E:/data/saliency/SALICON/Overfit/fine'
COARSE_PATH = 'E:/data/saliency/SALICON/Overfit/coarse'
LABEL_PATH = 'E:/data/saliency/SALICON/Overfit/label'
PARAM_PATH = 'E:/cv_toy/saliency/lwsal/mxnet/params'
MODEL_PREFIX = 'params/lwsp'

def normalize_255(arr):
	if np.min(arr == np.max(arr)):
		return np.clip(255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr) + 1e-7), 0, 255)
	else:
		return np.clip(255 * (arr - np.min(arr)) / float(np.max(arr) - np.min(arr)), 0, 255)

def squeez(arr):
	if np.mean(arr) < 45:
		return arr
	avg = np.mean(arr)
	arr = arr - avg
	return normalize_255(arr)

# def inference_batch():
# 	diter = SaliconIter(FINE_PATH, COARSE_PATH, LABEL_PATH, 48)
# 	symbol = net()
# 	_, arg_params, aux_params = mx.model.load_checkpoint('params/lwsp', 12)
# 	mod = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('fine', 'coarse'), label_names=('label', ))
# 	mod.bind(for_training=False, data_shapes=[('fine', (1,3,80,80)), ('coarse', (1,3,80,80))], label_shapes=[('label', (1,1,10,10)), ])
# 	mod.set_params(arg_params, aux_params, allow_missing=True)

# 	f = 'COCO_train2014_000000000110.jpg'
# 	prefix = f.split('.')[0]
# 	data_path = os.path.join('E:/data/saliency/SALICON/Overfit/', prefix + '.jpg')
# 	data_raw = cv2.imread(data_path)
# 	data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
# 	res = np.zeros((480//8, 640//8))
# 	raw = np.zeros((480, 640, 3))
# 	batch = next(diter)
# 	mod.forward(batch)
# 	ret = mod.get_outputs()[0].asnumpy()
# 	print(ret.shape)
# 	for i in range(6):
# 		for j in range(8):
# 			res[i*10:(i+1)*10, j*10:(j+1)*10] = ret[i*8+j][0].transpose()
# 			raw[i*80:(i+1)*80, j*80:(j+1)*80, :] = batch.data[0].asnumpy().transpose()[:,:,:,i*8+j]
# 	res = normalize_255(res)
# 	cv2.imwrite(os.path.join('output', prefix + '_mx.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
# 	cv2.imwrite(os.path.join('output', prefix + '_rw.jpg'), cv2.resize(raw.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))


def test_mit300():
	diter = SaliconIter(FINE_PATH, COARSE_PATH, LABEL_PATH, 48)
	symbol = net()
	_, arg_params, aux_params = mx.model.load_checkpoint('params/lwsp', 58)
	mod = mx.mod.Module(symbol=symbol, context=mx.gpu(0), data_names=('fine', 'coarse'), label_names=None)
	mod.bind(for_training=False, data_shapes=[('fine', (1,3,80,80)), ('coarse', (1,3,80,80))], label_shapes=mod._label_shapes)
	mod.set_params(arg_params, aux_params, allow_missing=True)
	dpath = 'E:/data/saliency/MIT1003/ALLSTIMULI'
	# dpath = 'E:/data/saliency/MIT300/BenchmarkIMAGES'
	# for _,_,fs in os.walk('E:/data/saliency/MIT300/BenchmarkIMAGES/'):
	for _,_,fs in os.walk(dpath):
		for f in fs[:500]:
			prefix = f.split('.')[0]
			data_path = os.path.join(dpath, prefix + '.jpeg')
			data_raw = cv2.imread(data_path)
			h_raw, w_raw, c  = data_raw.shape
			h_new, w_new = int(round(h_raw / 80) * 80), int(round(w_raw / 80) * 80)
			print('size: %s->%s, %s->%s' % (w_raw, w_new, h_raw, h_new))
			data_new = cv2.resize(data_raw, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)
			data_pad = cv2.copyMakeBorder(data_new, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
			res = np.zeros((h_new//8, w_new//8))
			res_c = np.zeros((h_new//8, w_new//8))
			res_f = np.zeros((h_new//8, w_new//8))
			for i in range(h_new//80):
				for j in range(w_new//80):
					crop_fine = np.expand_dims(np.swapaxes(data_new[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2), axis=0).astype(np.float32)
					crop_coarse = np.expand_dims(np.swapaxes(cv2.resize(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], (80, 80), interpolation=cv2.INTER_LANCZOS4), 0, 2), axis=0).astype(np.float32)
					batch = mx.io.DataBatch([mx.nd.array(crop_fine), mx.nd.array(crop_coarse)])
					mod.forward(batch)
					ret = mod.get_outputs()[1].asnumpy()[0][0].transpose()
					ret_c = mod.get_outputs()[2].asnumpy()[0][0].transpose()
					ret_f = mod.get_outputs()[3].asnumpy()[0][0].transpose()
					res[i*10:(i+1)*10, j*10:(j+1)*10] = ret
					res_c[i*10:(i+1)*10, j*10:(j+1)*10] = ret_c
					res_f[i*10:(i+1)*10, j*10:(j+1)*10] = ret_f
			print('[DEBUG] fine range: %s~%s, coarse range: %s~%s' % (np.min(res_f), np.max(res_f), np.min(res_c), np.max(res_c)))
			res = normalize_255(res)
			res = cv2.blur(res, (5,5))
			# res_c = normalize_255(res_c)
			# res_f = normalize_255(res_f)
			res = cv2.resize(res, (w_raw, h_raw), interpolation=cv2.INTER_LANCZOS4)
			res = cv2.blur(res, (25, 25))
			res = normalize_255(res)
			cv2.imwrite(os.path.join('output/mit1003', prefix + '_mx.jpg'), res.astype(np.uint8))
			# cv2.imwrite(os.path.join('output', prefix + '_mx.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			# cv2.imwrite(os.path.join('output', prefix + '_coarse.jpg'), cv2.resize(res_c.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			# cv2.imwrite(os.path.join('output', prefix + '_fine.jpg'), cv2.resize(res_f.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			cv2.imwrite(os.path.join('output/mit1003', prefix + '_rw.jpg'), data_raw)	

def inference():
	diter = SaliconIter(FINE_PATH, COARSE_PATH, LABEL_PATH, 48)
	symbol = net()
	_, arg_params, aux_params = mx.model.load_checkpoint('params/lwsp', 58)
	mod = mx.mod.Module(symbol=symbol, context=mx.gpu(0), data_names=('fine', 'coarse'), label_names=None)
	mod.bind(for_training=False, data_shapes=[('fine', (1,3,80,80)), ('coarse', (1,3,80,80))], label_shapes=mod._label_shapes)
	mod.set_params(arg_params, aux_params, allow_missing=True)

	for _,_,fs in os.walk('E:/data/saliency/SALICON/Valid/'):
		random.shuffle(fs)
		for f in fs[:500]:
			prefix = f.split('.')[0]
			data_path = os.path.join('E:/data/saliency/SALICON/Valid', prefix + '.jpg')
			data_lab = cv2.imread(os.path.join('E:/data/saliency/SALICON/ValidFix', prefix + '.jpeg'))
			data_raw = cv2.imread(data_path)
			data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
			res = np.zeros((480//8, 640//8))
			res_c = np.zeros((480//8, 640//8))
			res_f = np.zeros((480//8, 640//8))
			for i in range(6):
				for j in range(8):
					crop_fine = np.expand_dims(np.swapaxes(data_raw[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2), axis=0).astype(np.float32)
					crop_coarse = np.expand_dims(np.swapaxes(cv2.resize(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], (80, 80), interpolation=cv2.INTER_LANCZOS4), 0, 2), axis=0).astype(np.float32)
					batch = mx.io.DataBatch([mx.nd.array(crop_fine), mx.nd.array(crop_coarse)])
					mod.forward(batch)
					ret = mod.get_outputs()[1].asnumpy()[0][0].transpose()
					ret_c = mod.get_outputs()[2].asnumpy()[0][0].transpose()
					ret_f = mod.get_outputs()[3].asnumpy()[0][0].transpose()
					res[i*10:(i+1)*10, j*10:(j+1)*10] = ret
					res_c[i*10:(i+1)*10, j*10:(j+1)*10] = ret_c
					res_f[i*10:(i+1)*10, j*10:(j+1)*10] = ret_f
			print('[DEBUG] fine range: %s~%s, coarse range: %s~%s' % (np.min(res_f), np.max(res_f), np.min(res_c), np.max(res_c)))
			res = normalize_255(res)
			res = cv2.blur(res, (5,5))
			# res_c = normalize_255(res_c)
			# res_f = normalize_255(res_f)
			res = cv2.resize(res, (640, 480), interpolation=cv2.INTER_LANCZOS4)
			res = cv2.blur(res, (25, 25))
			res = normalize_255(res)
			res = squeez(res)
			cv2.imwrite(os.path.join('output/salicon', prefix + '_mx.jpg'), res.astype(np.uint8))
			# cv2.imwrite(os.path.join('output', prefix + '_mx.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			# cv2.imwrite(os.path.join('output', prefix + '_coarse.jpg'), cv2.resize(res_c.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			# cv2.imwrite(os.path.join('output', prefix + '_fine.jpg'), cv2.resize(res_f.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
			cv2.imwrite(os.path.join('output/salicon', prefix + '_lb.jpg'), data_lab)
			cv2.imwrite(os.path.join('output/salicon', prefix + '_rw.jpg'), data_raw)


if __name__ == '__main__':
	test_mit300()