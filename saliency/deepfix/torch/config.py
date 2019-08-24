import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_logger')

DATA_PATH = 'E:/Dataset/SALICON/Tiny/data/'
LABEL_PATH = 'E:/Dataset/SALICON/Tiny/label/'
PARAM_PATH = 'E:/Dataset/SALICON/Tiny/params/'
BIAS_PATH = '../bias/'
BATCH_SIZE = 1
EPOCHES = 1000
SAVE_STEP = 50
GRAD_ACCUM = 1