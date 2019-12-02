'''
ResNet Saliency a baseline model with ResNet 50 backbone

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torchvision.models.resnet import resnet50, Bottleneck
from thop import profile
from PIL import Image

class ResidualBlock(nn.Module):

    def __init__(self, inp, out, exp, droprate, keep_input_size=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.keeprate = droprate
        self.res_flag = inp == out
        # inp = int(inp * droprate) if not keep_input_size else inp
        # out = int(out * droprate)
        mid = int(out // exp)
        self.conv1 = nn.Conv2d(inp, int(mid*self.keeprate), kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(int(mid*self.keeprate))
        self.conv2 = nn.Conv2d(mid, int(mid*self.keeprate), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(mid*self.keeprate))
        self.conv3 = nn.Conv2d(mid, int(out*self.keeprate), kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(int(out*self.keeprate))
        self.relu = nn.ReLU(inplace=True)
        if not self.res_flag:
            self.convr = nn.Conv2d(inp, int(out*self.keeprate), kernel_size=1, stride=stride, padding=0, bias=False)
            self.bnr = nn.BatchNorm2d(int(out*self.keeprate))
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        # nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.kaiming_normal_(self.conv3.weight)
        # nn.init.constant_(self.conv3.bias, 0.0)
        if not self.res_flag:
            nn.init.kaiming_normal_(self.convr.weight)
            # nn.init.constant_(self.convr.bias, 0.0)

    def init_pretrained(self, conv1, conv2, conv3, convr=None):
        idx1 = np.sum(conv1.cpu().numpy(), axis=(1,2,3))[::-1][:mid]
        idx2 = np.sum(conv2.cpu().numpy(), axis=(1,2,3))[::-1][:mid]
        idx3 = np.sum(conv3.cpu().numpy(), axis=(1,2,3))[::-1][:out]
        self.conv1.weight.data = conv1.weight.data[idx1.tolist()].clone()        
        self.conv2.weight.data = conv2.weight.data[idx2.tolist()].clone()        
        self.conv3.weight.data = conv3.weight.data[idx3.tolist()].clone()        
        if not self.res_flag and convr != None:
            idxr = np.sum(convr.cpu().numpy(), axis=(1,2,3))[::-1][:out]
            self.convr.weight.data = convr.weight.data[idxr.tolist()].clone()    

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

    def __init__(self, inp, out, droprate):
        super(ScaleUpBlock, self).__init__()
        self.keeprate = droprate
        self.res_flag = inp == out
        # inp = int(inp * droprate) if inp != 3 else 3
        # out = int(out * droprate)
        self.deconv1 = nn.ConvTranspose2d(inp, inp, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inp, int(out*self.keeprate), kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out*self.keeprate))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out, int(out*self.keeprate), kernel_size=1, stride=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.deconv1.weight)
        # nn.init.constant_(self.deconv1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.kaiming_normal_(self.conv3.weight)
        # nn.init.constant_(self.conv3.bias, 0.0)

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

    def __init__(self, ):
        super(Model, self).__init__()
        self.keeprate = 0.5
        modules_raw = list(resnet50(pretrained=True).children())[:-2]
        modules_flat = [
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ResidualBlock(64, 256, 4, self.keeprate, keep_input_size=True),
            ResidualBlock(256, 256, 4, self.keeprate),
            ResidualBlock(256, 256, 4, self.keeprate),
            ResidualBlock(256, 512, 4, self.keeprate, stride=2),
            ResidualBlock(512, 512, 4, self.keeprate),
            ResidualBlock(512, 512, 4, self.keeprate),
            ResidualBlock(512, 512, 4, self.keeprate),
            ResidualBlock(512, 1024, 4, self.keeprate, stride=2),
            ResidualBlock(1024, 1024, 4, self.keeprate),
            ResidualBlock(1024, 1024, 4, self.keeprate),
            ResidualBlock(1024, 1024, 4, self.keeprate),
            ResidualBlock(1024, 1024, 4, self.keeprate),
            ResidualBlock(1024, 1024, 4, self.keeprate),
            ResidualBlock(1024, 2048, 4, self.keeprate, stride=2),
            ResidualBlock(2048, 2048, 4, self.keeprate),
            ResidualBlock(2048, 2048, 4, self.keeprate),
        ]
        conv_list1 = [x for x in nn.Sequential(*modules_raw).modules() if type(x) == nn.Conv2d or type(x) == nn.ConvTranspose2d]
        conv_list2 = [x for x in nn.Sequential(*modules_flat).modules() if type(x) == nn.Conv2d or type(x) == nn.ConvTranspose2d]
        # for x in conv_list1:
        #     print(x)
        # print('=========================')
        # for x in conv_list2:
        #     print(x)
        bn_list1 = [x for x in nn.Sequential(*modules_raw).modules() if type(x) == nn.BatchNorm2d]
        bn_list2 = [x for x in nn.Sequential(*modules_flat).modules() if type(x) == nn.BatchNorm2d]
        try:
            assert len(conv_list1) == len(conv_list2) and len(conv_list1) == len(bn_list1) and len(bn_list1) == len(bn_list2)
        except Exception as e:
            print('ASSERT ERROR: %s-%s' % (len(conv_list1), len(conv_list2)))
            exit()
        for i in range(len(conv_list1)):
            raw, flat, raw_bn, flat_bn = conv_list1[i], conv_list2[i], bn_list1[i], bn_list2[i]
            s1 = flat.weight.shape
            weight = raw.weight.data.clone()
            weight_bn = raw_bn.weight.data.clone()
            index = np.argsort(np.sum(raw.weight.data.abs().clone().numpy(), axis=(1,2,3)))
            if i != 0:
                index = index[:int(len(index)*self.keeprate)]
            flat.weight.data = weight[index]
            flat_bn.weight.data = weight_bn[index]
            s2 = flat.weight.shape
            print('%s-->%s' % (str(s1), str(s2)))
            # print('%s-->%s' % (str(flat), str(raw)))

        self.decoder1 = ScaleUpBlock(2048, 1024, self.keeprate)
        self.decoder2 = ScaleUpBlock(1024, 512, self.keeprate)
        self.decoder3 = ScaleUpBlock(512, 256, self.keeprate)
        self.encode_image = nn.Sequential(*modules_flat)
        # for x in self.encode_image.modules():
        #     if type(x)==nn.Conv2d or type(x)==nn.BatchNorm2d:
        #         print(x)
        # exit()
        self.saliency = nn.Conv2d(int(256*self.keeprate), 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.__init_weights__()

    def __init_weights__(self):
        nn.init.kaiming_normal_(self.saliency.weight)
        # nn.init.constant_(self.saliency.bias, 0.0)
        # nn.init.constant_(self.saliency.weight, 1.)

    def forward(self, x):
        x = self.encode_image(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.saliency(x)
        x = F.relu(x, inplace=True)
        return x

if __name__ == "__main__":
    img_path = 'G:\\datasets\\saliency\\SALICON\\images\\tiny\\i1.jpg'
    img = np.array(Image.open(img_path).resize((320, 256))).swapaxes(0,2).swapaxes(1,2)[np.newaxis]
    img = Variable(torch.from_numpy(img)).type(torch.FloatTensor)
    model = Model().cuda()
    out = model(img.cuda())
    print('output shape: %s' % (str(out.shape)))
    out = out.cpu().data.numpy()[0][0]
    out = 255. * (out - np.min(out)) / (np.max(out) - np.min(out))
    out = Image.fromarray(out.astype('uint8')).resize((640, 480)).show()

    # flops, params = profile(model.cuda(), inputs=(sample_input,))
    # g_flops = flops / float(1024 * 1024 * 1024)
    # m_params = params / float(1024 * 1024)
    # line_1 = 'proned[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnetsal', round(g_flops, 4), round(m_params, 4))
    # print(line_1)

    # 1.0 GFLOPs=39.1216G   paramsize=68.367M
    # 0.5 GFLOPs=9.0962G    paramsize=13.9752M
    # 0.25 GFLOPs=2.3563G    paramsize=3.2425M