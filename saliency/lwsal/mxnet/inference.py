import mxnet
import cv2

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

def inference():
	diter = SaliconIter(FINE_PATH, COARSE_PATH, LABEL_PATH, 48)
	symbol = overfit()
	_, arg_params, aux_params = mx.model.load_checkpoint('params/lwsp', 500)
	mod = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('fine', 'coarse'), label_names=None)
	mod.bind(for_training=False, data_shapes=[('fine', (1,3,80,80)), ('coarse', (1,3,80,80))], label_shapes=mod._label_shapes)
	mod.set_params(arg_params, aux_params, allow_missing=True)

	f = 'COCO_train2014_000000000110.jpg'
	prefix = f.split('.')[0]
	data_path = os.path.join('E:/data/saliency/SALICON/Overfit/', prefix + '.jpg')
	data_raw = cv2.imread(data_path)
	data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
	res = np.zeros((480//8, 640//8))
	for i in range(6):
		for j in range(8):
			# crop_fine = np.expand_dims(np.swapaxes(data_raw[i*80:(i+1)*80, j*80:(j+1)*80], 0, 2), axis=0).astype(np.float32)
			# crop_coarse = np.expand_dims(np.swapaxes(cv2.resize(data_pad[i*80:(i+3)*80, j*80:(j+3)*80], (80, 80), interpolation=cv2.INTER_LANCZOS4), 0, 2), axis=0).astype(np.float32)
			# batch = mx.io.DataBatch([mx.nd.array(crop_fine), mx.nd.array(crop_coarse)])
			mod.forward(batch)
			ret = mod.get_outputs()[0].asnumpy()[0][0]
			res[i*10:(i+1)*10, j*10:(j+1)*10] = ret
	res = normalize_255(res)
	cv2.imwrite(os.path.join('output', prefix + '_mx.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))

def inference_batch():
	diter = SaliconIter(FINE_PATH, COARSE_PATH, LABEL_PATH, 48)
	symbol = net()
	_, arg_params, aux_params = mx.model.load_checkpoint('params/lwsp', 1000)
	mod = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('fine', 'coarse'), label_names=('label', ))
	mod.bind(for_training=False, data_shapes=[('fine', (1,3,80,80)), ('coarse', (1,3,80,80))], label_shapes=[('label', (1,1,10,10)), ])
	mod.set_params(arg_params, aux_params, allow_missing=True)

	f = 'COCO_train2014_000000000110.jpg'
	prefix = f.split('.')[0]
	data_path = os.path.join('E:/data/saliency/SALICON/Overfit/', prefix + '.jpg')
	data_raw = cv2.imread(data_path)
	data_pad = cv2.copyMakeBorder(data_raw, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
	res = np.zeros((480//8, 640//8))
	raw = np.zeros((480, 640, 3))
	batch = next(diter)
	mod.forward(batch)
	ret = mod.get_outputs()[0].asnumpy()
	print(ret.shape)
	for i in range(6):
		for j in range(8):
			res[i*10:(i+1)*10, j*10:(j+1)*10] = ret[i*8+j][0].transpose()
			raw[i*80:(i+1)*80, j*80:(j+1)*80, :] = batch.data[0].asnumpy().transpose()[:,:,:,i*8+j]
	res = normalize_255(res)
	cv2.imwrite(os.path.join('output', prefix + '_mx.jpg'), cv2.resize(res.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))
	cv2.imwrite(os.path.join('output', prefix + '_rw.jpg'), cv2.resize(raw.astype(np.uint8), (640, 480), interpolation=cv2.INTER_LANCZOS4))


if __name__ == '__main__':
	inference_batch()