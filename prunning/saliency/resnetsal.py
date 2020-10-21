'''
ResNet Saliency a baseline model with ResNet 50 backbone

@author: Hamed R. Tavakoli

To perform prunning, first train this model using train.py, then provide the checkpoint and keep rate (1 - prune rate).

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

from torch.autograd import Variable
from torchvision.models.resnet import resnet50, Bottleneck
from thop import profile
from PIL import Image

class ResidualBlock(nn.Module):

    def __init__(self, inp, out, exp, dropend, keeprate, keep_input_size=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.keeprate = keeprate
        self.dropend = dropend
        self.res_flag = inp == out
        inp = int(inp * (keeprate if dropend == 'head' else 1)) if not keep_input_size else inp
        out = int(out * (keeprate if dropend == 'tail' else 1))
        mid = int(out // exp)
        self.conv1 = nn.Conv2d(inp, mid, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        if not self.res_flag:
            self.convr = nn.Conv2d(inp, out, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bnr = nn.BatchNorm2d(out)
        self.__init_weight__()

    def __init_weight__(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        if not self.res_flag:
            nn.init.kaiming_normal_(self.convr.weight)

    def init_pretrained(self, resblock):
        idxc, idxb = 0, 0
        for src in resblock.modules():
            if type(src) == nn.Conv2d:
                if idxc == 0:
                    des = self.conv1
                elif idxc == 1:
                    des = self.conv2
                elif idxc == 2:
                    des = self.conv3
                elif idxc == 3:
                    des = self.convr
                c1, c2 = des.weight.data.shape[0], des.weight.data.shape[1]
                d1 = np.squeeze(np.argwhere(np.argsort(np.sum(np.absolute(src.weight.data.cpu().numpy()), axis=(1,2,3)))[::-1][:c1] + 1.))
                d2 = np.squeeze(np.argwhere(np.argsort(np.sum(np.absolute(src.weight.data.cpu().numpy()), axis=(0,2,3)))[::-1][:c2] + 1.))
                des.weight.data = src.weight.data[d1.tolist(),:,:,:][:,d2.tolist(),:,:].clone()
                idxc += 1
            if type(src) == nn.BatchNorm2d:
                if idxb == 0:
                    des = self.bn1
                elif idxb == 1:
                    des = self.bn2
                elif idxb == 2:
                    des = self.bn3
                elif idxb == 3:
                    des = self.bnr
                c1 = des.weight.data.shape[0]
                d1 = np.squeeze(np.argwhere(np.argsort(np.absolute(src.weight.data.cpu().numpy()))[::-1][:c1] + 1.))
                des.weight.data = src.weight.data[d1.tolist()].clone()
                des.bias.data = src.bias.data[d1.tolist()].clone()
                des.running_mean = src.running_mean[d1.tolist()].clone()
                des.running_var = src.running_var[d1.tolist()].clone()
                idxb += 1

    def forward(self, x):
        residual = x
        if not self.res_flag:
            residual = self.convr(residual)
            residual = self.bnr(residual)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if not self.res_flag:
            x += residual
        x = self.relu(x)
        return x

class ScaleUpBlock(nn.Module):

    def __init__(self, inp, out, dropend, keeprate):
        super(ScaleUpBlock, self).__init__()
        self.keeprate = keeprate
        self.res_flag = inp == out
        inp = int(inp * (keeprate if dropend == 'head' else 1)) if inp != 3 else 3
        out = int(out * (keeprate if dropend == 'tail' else 1))
        self.deconv1 = nn.ConvTranspose2d(inp, inp, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out, out, kernel_size=1, stride=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.__init_weight__()

    def __init_weight__(self):
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)

    def init_pretrained(self, scaleupblock):
        idxc, idxb = 0, 0
        for src in scaleupblock.modules():
            if type(src) == nn.Conv2d or type(src) == nn.ConvTranspose2d:
                if idxc == 0:
                    des = self.deconv1
                elif idxc == 1:
                    des = self.conv2
                elif idxc == 2:
                    des = self.conv3
                c1, c2 = des.weight.data.shape[0], des.weight.data.shape[1]
                d1 = np.squeeze(np.argwhere(np.argsort(np.sum(np.absolute(src.weight.data.cpu().numpy()), axis=(1,2,3)))[::-1][:c1] + 1.))
                d2 = np.squeeze(np.argwhere(np.argsort(np.sum(np.absolute(src.weight.data.cpu().numpy()), axis=(0,2,3)))[::-1][:c2] + 1.))
                des.weight.data = src.weight.data[d1.tolist(),:,:,:][:,d2.tolist(),:,:].clone()
                idxc += 1
            if type(src) == nn.BatchNorm2d:
                if idxb == 0:
                    des = self.bn1
                elif idxb == 1:
                    des = self.bn2
                c1 = des.weight.data.shape[0]
                d1 = np.squeeze(np.argwhere(np.argsort(np.absolute(src.weight.data.cpu().numpy()))[::-1][:c1] + 1.))
                des.weight.data = src.weight.data[d1.tolist()].clone()
                des.bias.data = src.bias.data[d1.tolist()].clone()
                des.running_mean = src.running_mean[d1.tolist()].clone()
                des.running_var = src.running_var[d1.tolist()].clone()
                idxb += 1

    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x


class Model(nn.Module):

    def __init__(self, keeprate=1.):
        super(Model, self).__init__()
        self.keeprate = keeprate
        modules_flat = [
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ResidualBlock(64, 256, 4, 'tail', self.keeprate, keep_input_size=True),
            ResidualBlock(256, 256, 4, 'head', self.keeprate),
            ResidualBlock(256, 256, 4, 'tail', self.keeprate),
            ResidualBlock(256, 512, 4, 'head', self.keeprate, stride=2),
            ResidualBlock(512, 512, 4, 'tail', self.keeprate),
            ResidualBlock(512, 512, 4, 'head', self.keeprate),
            ResidualBlock(512, 512, 4, 'tail', self.keeprate),
            ResidualBlock(512, 1024, 4, 'head', self.keeprate, stride=2),
            ResidualBlock(1024, 1024, 4, 'tail', self.keeprate),
            ResidualBlock(1024, 1024, 4, 'head', self.keeprate),
            ResidualBlock(1024, 1024, 4, 'tail', self.keeprate),
            ResidualBlock(1024, 1024, 4, 'head', self.keeprate),
            ResidualBlock(1024, 1024, 4, 'tail', self.keeprate),
            ResidualBlock(1024, 2048, 4, 'head', self.keeprate, stride=2),
            ResidualBlock(2048, 2048, 4, 'tail', self.keeprate),
            ResidualBlock(2048, 2048, 4, 'head', self.keeprate),
            ScaleUpBlock(2048, 1024, 'tail', self.keeprate),
            ScaleUpBlock(1024, 512, 'head', self.keeprate),
            ScaleUpBlock(512, 256, 'tail', self.keeprate),
        ]
        self.encode_image = nn.Sequential(*modules_flat)
        self.saliency = nn.Conv2d(int(256*self.keeprate), 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.____init_weight__s__()

    def __repr__(self):
        return 'ResNet model object: ' + str(id(self))

    def ____init_weight__s__(self):
        nn.init.kaiming_normal_(self.saliency.weight)

    def forward(self, x):
        x = self.encode_image(x)
        x = self.saliency(x)
        x = F.relu(x, inplace=True)
        return x

    def prune(self, prunerate=1.):
        newmodel = Model(prunerate)
        idx = 0
        for x, y in zip(self.encode_image.modules(), newmodel.encode_image.modules()):
            if type(x) in [nn.Conv2d, nn.BatchNorm2d] and idx == 0:
                y.weight.data = x.weight.data.clone()
            elif type(x) == ResidualBlock:
                y.init_pretrained(x)
                idx += 1
            elif type(x) == ScaleUpBlock:
                y.init_pretrained(x)
                idx += 1
        cs = newmodel.saliency.weight.data.shape[1]
        ds = np.squeeze(np.argwhere(np.argsort(np.sum(np.absolute(self.saliency.weight.data.cpu().numpy()), axis=(0,2,3)))[::-1][:cs] + 1.))
        newmodel.saliency.weight.data = self.saliency.weight.data[:,ds.tolist(),:,:].clone()
        return newmodel

def init_pretrained():
    checkpoint = torch.load('')
    state_dict = checkpoint['state_dict']

    

def test_prune():
    # init trained checkpoints
    checkpoint = torch.load('G:\\checkpoints\\saliency\\resnetprune\\model_best_256x320.pth.tar')
    state_dict = checkpoint['state_dict']

    # init test image
    img_path = 'G:\\datasets\\saliency\\SALICON\\images\\tiny\\COCO_train2014_000000000110.jpg'
    img = np.array(Image.open(img_path).resize((320, 256))).swapaxes(0,2).swapaxes(1,2)[np.newaxis]
    img = Variable(torch.from_numpy(img)).type(torch.FloatTensor).cuda()

    # init raw model
    model = Model().cuda()
    model.load_state_dict(state_dict=state_dict, strict=True)

    # init pruned model. the parameter @keeprate does not acturally keep this much, because the residual module usually in a three layers form, 
    # the dropping rate is asymmetric between different modules. the actual keeprate is higher than the parameter
    newmodel = model.prune(0.5).cuda()

    # show output of raw model
    out = model(img.cuda())
    print('output shape: %s' % (str(out.shape)))
    out = out.cpu().data.numpy()[0][0]
    out = 255. * (out - np.min(out)) / (np.max(out) - np.min(out))
    out = Image.fromarray(out.astype('uint8')).resize((640, 480)).show()

    # show output of pruned model
    out = newmodel(img.cuda())
    print('output shape: %s' % (str(out.shape)))
    out = out.cpu().data.numpy()[0][0]
    out = 255. * (out - np.min(out)) / (np.max(out) - np.min(out))
    out = Image.fromarray(out.astype('uint8')).resize((640, 480)).show()

    # show gflops of raw model
    flops, params = profile(model.cuda(), inputs=(img,))
    g_flops = flops / float(1024 * 1024 * 1024)
    m_params = params / float(1024 * 1024)
    line_1 = 'proned[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnetsal', round(g_flops, 4), round(m_params, 4))
    
    # show gflops of pruned model
    flops, params = profile(newmodel.cuda(), inputs=(img,))
    g_flops = flops / float(1024 * 1024 * 1024)
    m_params = params / float(1024 * 1024)
    line_2 = 'proned[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnetsal', round(g_flops, 4), round(m_params, 4))

    print(line_1)
    print(line_2)

if __name__ == "__main__":
    test_prune()