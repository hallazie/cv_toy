'''
ResNet Saliency a baseline model with ResNet 50 backbone

@author: Hamed R. Tavakoli
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

from thop import profile

class RusidualBlock(nn.Module):

    def __init__(self, inp, out, exp, droprate):
        super(RusidualBlock, self).__init__()
        self.res_flag = inp == out
        inp = int(inp * droprate) if inp != 3 else 3
        out = int(out * droprate)
        mid = int((out * droprate) // exp)
        self.conv1 = nn.Conv2d(inp, mid, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.convr = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.bnr = nn.BatchNorm2d(out)

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

class _ScaleUp(nn.Module):

    def __init__(self, in_size, out_size):
        super(_ScaleUp, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_size),
            nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )

        self.__init_weights__()

    def __init_weights__(self):

        nn.init.kaiming_normal_(self.conv[0].weight)
        nn.init.constant_(self.conv[0].bias, 0.0)
        nn.init.kaiming_normal_(self.conv[3].weight)
        nn.init.constant_(self.conv[3].bias, 0.0)

        nn.init.kaiming_normal_(self.up[0].weight)
        nn.init.constant_(self.up[0].bias, 0.0)

    def forward(self, inputs):
        output = self.up(inputs)
        output = self.conv(output)
        return output

def flat_bottleneck(bottleneck):
    for x in bottleneck.modules():
        print(x)
    print('----------')

class Model(nn.Module):

    def __init__(self, ):
        super(Model, self).__init__()
        # self.encode_image = resnet50(pretrained=True)
        # modules = list(self.encode_image.children())[:-2]
        modules_flat = [
            RusidualBlock(3, 32, 4, 0.5),
            RusidualBlock(32, 32, 4, 0.5),
        ]
        self.encode_image = nn.Sequential(*modules_flat)

        for m in self.encode_image.modules():
            if type(m) == RusidualBlock:
                flat_bottleneck(m)
        
        self.decoder1 = _ScaleUp(2048, 1024)
        self.decoder2 = _ScaleUp(1024, 512)
        self.decoder3 = _ScaleUp(512, 256)
        self.saliency = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.__init_weights__()

    def __init_weights__(self):

        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)

    def forward(self, x):
        x1 = self.encode_image(x)
        x1 = self.decoder1(x1)
        x1 = self.decoder2(x1)
        x1 = self.decoder3(x1)
        sal = self.saliency(x1)
        sal = F.relu(x1, inplace=True)
        return sal

if __name__ == "__main__":
    sample_input = torch.zeros(1, 3, 256, 320).cuda()
    model = Model().cuda()
    model(sample_input)
    flops, params = profile(model.cuda(), inputs=(sample_input,))
    g_flops = flops / float(1024 * 1024 * 1024)
    m_params = params / float(1024 * 1024)
    line_1 = 'proned[%s]\tGFLOPs=%sG\tparamsize=%sM\n' % ('resnetsal', round(g_flops, 4), round(m_params, 4))
    print(line_1)
