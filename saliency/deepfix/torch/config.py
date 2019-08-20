import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_logger')

DATA_PATH = '/Users/xiaoshanghua/Workspace/cv_toy/saliency/data/data/'
LABEL_PATH = '/Users/xiaoshanghua/Workspace/cv_toy/saliency/data/label/'
PARAM_PATH = '../params'
BATCH_SIZE = 1
EPOCHES = 1000
SAVE_STEP = 200
GRAD_ACCUM = 1