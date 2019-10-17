
'''
Get the model of interest

@author: Hamed R. Tavakoli
'''

MODEL_NAME = ['deepgaze2', 'mlnet', 'resnetsal', 'salicon', 'samres', 'deepfix', 'fastsal', 'mobilesal']


class ModelConfig:

    MODEL = 'samres'
    B_SIZE = 1
    N_STEP = 3
    W_OUT = 40
    H_OUT = 30
    W_IN = 320
    H_IN = 256


def make_model(meta_config):
    ''' make the model from the meta config '''

    m = __import__("models.{}".format(meta_config.MODEL))
    m = getattr(m, meta_config.MODEL)
    model = getattr(m, 'Model')

    if meta_config.MODEL == 'samres':
        object = model(meta_config.B_SIZE, meta_config.N_STEP, meta_config.W_OUT, meta_config.H_OUT)
    else:
        object = model()

    return object


if __name__ == '__main__':
    meta_config = ModelConfig
    a = make_model(meta_config)
    print(a)
