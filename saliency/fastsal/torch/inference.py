'''
A sample script to show how to use the saliency model

@author: Hamed R. Tavakoli
'''
import re
import torch

import os
import time
import numpy as np

import torchvision.transforms as transforms

from PIL import Image

from scipy.ndimage.filters import gaussian_filter

from models import make_model, ModelConfig, MODEL_NAME
from utils import padded_resize, postprocess_predictions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.device(device)

def _normalize_data(x):
    x = x.view(x.size(0), -1)
    x_max, idx = torch.max(x, dim=1, keepdim=True)
    x = x / (x_max.expand_as(x) + 1e-8)
    return x


class EstimateSaliency(object):

    def __init__(self, img_path, model_cfg, model_path):
        super(EstimateSaliency, self).__init__()
        self.impath = img_path

        self.cfg = model_cfg

        print("Estimating: {}".format(self.cfg.MODEL))
        self.model = make_model(self.cfg).to(device)

        self.model.eval()
        self.load_checkpoint(os.path.join(model_path, self.cfg.MODEL, "model_best_{}x{}.pth.tar".format(self.cfg.H_IN, self.cfg.W_IN)))

    def load_checkpoint(self, model_path):
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)

            # '.'s are no longer allowed in module names, but pervious _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            self.model.load_state_dict(state_dict=state_dict, strict=True)
            print("=> loaded checkpoint '{}' )".format(model_path))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def estimate(self, savefolder):

        output_path = savefolder
        new_path = self.impath
        time_list = []
        for file in os.listdir(new_path):

            if not file.endswith('.jpg'):
                continue

            imageName = file[:-4]
            imgO = Image.open(os.path.join(self.impath, file)).convert('RGB')
            orig_w = imgO.size[1]
            orig_h = imgO.size[0]

            imgO = padded_resize(imgO, self.cfg.H_IN, self.cfg.W_IN)

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

            img1 = transform(imgO)

            img1 = img1.to(device)

            img1 = torch.unsqueeze(img1, dim=0)

            start = time.time()
            saloutput = self.model(img1)
            end = time.time()
            time_list.append(end-start)
            saloutput = _normalize_data(saloutput)
            saloutput = saloutput.view([1, self.cfg.H_OUT, self.cfg.W_OUT])
            saloutput = torch.squeeze(saloutput, 0)
            saloutput = saloutput.cpu().data.numpy()


            #saloutput = np.squeeze(saloutput, axis=0)
            saloutput = postprocess_predictions(saloutput, orig_w, orig_h)
            print('max-min:%s, %s' % (np.max(saloutput), np.min(saloutput)))

            a = 0.015*min(orig_w, orig_h)
            saloutput = (saloutput - np.min(saloutput)) / (np.max(saloutput) - np.min(saloutput))
            saloutput = gaussian_filter(saloutput, sigma=a)
            #saloutput = np.power(saloutput, 1.5)
            saloutput = (saloutput - np.min(saloutput)) / (np.max(saloutput) - np.min(saloutput))
            imgN = Image.fromarray((saloutput*255).astype(np.uint8))
            imgN.save('{}/{}.jpg'.format(output_path, imageName), 'JPEG')
            #imgN.save('{}/{}.png'.format(output_path, imageName), 'PNG')
        mean_runtime=np.mean(time_list)
        print("average running time:{}".format(mean_runtime))


# MLNet
#height_dim = 480
#width_dim = 640
#deeogaze
#height_dim = 240
#width_dim = 320
#SALICON
#height_dim = 600
#width_dim = 800
#ts = (64, 80) # resnet sal
#ts = (30, 40) # mlnet
#ts = (15, 20) # deep gaze
#ts = (37, 50) # salicon
#ts = (60, 80)#mlnet
#deep fix
#height_dim = 256
#width_dim = 320
#ts = (27, 35) deepfix
if __name__ == "__main__":

    folder = 'E:\\Dataset\\SALICON\\Tiny\\images\\val\\'
    res_folder = 'E:\\Dataset\\SALICON\\Tiny\\result\\'

    modelcfg = ModelConfig()
    modelcfg.MODEL = MODEL_NAME[7]
    modelcfg.H_IN = 256
    modelcfg.W_IN = 320
    modelcfg.H_OUT = 78
    modelcfg.W_OUT = 94
    model_trainer = EstimateSaliency(img_path=folder, model_cfg=modelcfg,
                                     model_path='E:\\Dataset\\SALICON\\Tiny\\output\\')

    model_trainer.estimate(savefolder=res_folder)