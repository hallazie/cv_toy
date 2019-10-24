"""
Script to train the models

@author: Hamed R. Tavakoli
"""

import os
import sys
import traceback
import torch

from models import make_model, ModelConfig, MODEL_NAME
from thop import profile

# mobile sal
height_dim = 256
width_dim = 320

ts = (32, 40) # fastsal
# ts = (78, 94)  # mobilesal

def info():
    cfg = ModelConfig()
    inp = torch.randn(1,3,256,320)
    lst = []
    for name in MODEL_NAME:
        try:
            cfg.MODEL = name
            model = make_model(cfg)
            flops, params = profile(model, inputs=(inp, ))
            g_flops = flops / float(1024*1024*1024)
            m_params = params / float(1024*1024)
            line = '[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % (name, round(g_flops, 4), round(m_params, 4))
            lst.append(line)
        except Exception as e:
            traceback.print_exc()
    with open('gflops.txt', 'w') as f:
        for l in lst:
            f.write(l)

if __name__ == "__main__":
    info()