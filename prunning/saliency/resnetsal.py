'''
ResNet Saliency a baseline model with ResNet 50 backbone

@author: Hamed R. Tavakoli
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

from thop import profile

class ResidualBlock(nn.Module):

    def __init__(self, inp, out, exp, droprate, keep_input_size=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.droprate = droprate
        self.res_flag = inp == out
        inp = int(inp * droprate) if not keep_input_size else inp
        out = int(out * droprate)
        mid = int((out * droprate) // exp)
        self.conv1 = nn.Conv2d(inp, mid, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.convr = nn.Conv2d(inp, out, kernel_size=1, stride=stride, padding=0)
        self.bnr = nn.BatchNorm2d(out)
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0.0)
        nn.init.kaiming_normal_(self.convr.weight)
        nn.init.constant_(self.convr.bias, 0.0)

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
        x += residual
        x = self.relu(x)
        return x

class ScaleUpBlock(nn.Module):

    def __init__(self, inp, out, droprate):
        super(ScaleUpBlock, self).__init__()
        self.droprate = droprate
        self.res_flag = inp == out
        inp = int(inp * droprate) if inp != 3 else 3
        out = int(out * droprate)
        self.deconv1 = nn.ConvTranspose2d(inp, inp, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out, out, kernel_size=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.constant_(self.deconv1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0.0)

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
        self.droprate = 0.5
        modules_raw = list(resnet50(pretrained=True).children())[:-2]
        modules_flat = [
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ResidualBlock(64, 256, 4, self.droprate, keep_input_size=True),
            ResidualBlock(256, 256, 4, self.droprate),
            ResidualBlock(256, 256, 4, self.droprate),
            ResidualBlock(256, 512, 4, self.droprate, stride=2),
            ResidualBlock(512, 512, 4, self.droprate),
            ResidualBlock(512, 512, 4, self.droprate),
            ResidualBlock(512, 512, 4, self.droprate),
            ResidualBlock(512, 1024, 4, self.droprate, stride=2),
            ResidualBlock(1024, 1024, 4, self.droprate),
            ResidualBlock(1024, 1024, 4, self.droprate),
            ResidualBlock(1024, 1024, 4, self.droprate),
            ResidualBlock(1024, 1024, 4, self.droprate),
            ResidualBlock(1024, 1024, 4, self.droprate),
            ResidualBlock(1024, 2048, 4, self.droprate, stride=2),
            ResidualBlock(2048, 2048, 4, self.droprate),
            ResidualBlock(2048, 2048, 4, self.droprate),
        ]
        conv_list1 = [repr(x) for x in nn.Sequential(*modules_raw).modules() if type(x) == nn.Conv2d]
        conv_list2 = [repr(x) for x in nn.Sequential(*modules_flat).modules() if type(x) == nn.Conv2d]
        for x in conv_list1:
            print(x)
        print('=====================')
        for x in conv_list2:
            print(x)
        assert len(conv_list1) == len(conv_list2)
        for x in zip(conv_list1, conv_list2):
            print('%s-----%s' % (x[0], x[1]))
        self.decoder1 = ScaleUpBlock(2048, 1024, self.droprate),
        self.decoder2 = ScaleUpBlock(1024, 512, self.droprate),
        self.decoder3 = ScaleUpBlock(512, 256, self.droprate),
        self.encode_image = nn.Sequential(*modules_flat)
        self.saliency = nn.Conv2d(int(256*self.droprate), 1, kernel_size=1, stride=1, padding=0)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)

    def forward(self, x):
        x = self.encode_image(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.saliency(x)
        x = F.relu(x, inplace=True)
        return x

if __name__ == "__main__":
    sample_input = torch.zeros(1, 3, 256, 320).cuda()
    model = Model().cuda()
    model(sample_input)
    flops, params = profile(model.cuda(), inputs=(sample_input,))
    g_flops = flops / float(1024 * 1024 * 1024)
    m_params = params / float(1024 * 1024)
    line_1 = 'proned[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnetsal', round(g_flops, 4), round(m_params, 4))
    print(line_1)
