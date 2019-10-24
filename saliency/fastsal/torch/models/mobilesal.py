import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v2


class MobileScaleUpV2(nn.Module):

    def __init__(self, in_size, out_size):
        super(MobileScaleUpV2, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2, groups=in_size, bias=False),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_size, in_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=2, groups=out_size, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.__init_weights__()

    def __init_weights__(self):

        nn.init.kaiming_normal_(self.conv[0].weight)
        nn.init.kaiming_normal_(self.conv[3].weight)
        nn.init.kaiming_normal_(self.conv[6].weight)
        nn.init.kaiming_normal_(self.up[0].weight)
        nn.init.kaiming_normal_(self.up[3].weight)

    def forward(self, inputs):
        output = self.up(inputs)
        output = self.conv(output)
        return output


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.encode_image = mobilenet_v2(pretrained=False).features

        self.decoder1 = MobileScaleUpV2(1280, 640)
        self.decoder2 = MobileScaleUpV2(640, 320)
        self.decoder3 = MobileScaleUpV2(320, 160)

        self.saliency = nn.Conv2d(160, 1, kernel_size=1, stride=1, padding=0)

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
        sal = F.relu(sal, inplace=True)
        return sal


if __name__=="__main__":
    sample_input = torch.ones(1, 3, 256, 320)
    model = Model()
    model.train()
    n_param = sum(p.numel() for p in model.parameters())
    print(model(sample_input).shape)
    print(n_param)

#if __name__=="__main__":
#    sample_input = torch.ones(1, 3, 256, 320)
#    model = MobileSalV2()
#    model.train()
#    n_param = sum(p.numel() for p in model.parameters())
#    print(model(sample_input).shape)
#    print(n_param)