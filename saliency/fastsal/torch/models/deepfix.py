'''
Implementation of Deepfix by Kruthiventi et al.
@author: Xiao Shanghua

Bug fixes, modified to support pre-trained model intitialization from VGG weights, removing bias for consistency
with experiment setup, etc. (See deepfix branch for original Shanghua's implementation). Hamed R. Tavakoli

'''

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url




class InceptionModule(nn.Module):
    def __init__(self, c_in):
        super(InceptionModule, self).__init__()
        self.seq_1 = nn.Sequential(
            self.Conv(c_in, 128, 1, 1, 1)
        )
        self.seq_2 = nn.Sequential(
            self.Conv(c_in, 128, 1, 1, 1),
            self.Conv(128, 256, 3, 1, 1)
        )
        self.seq_3 = nn.Sequential(
            self.Conv(c_in, 32, 1, 1, 1),
            self.Conv(32, 64, 3, 2, 2)
        )
        self.seq_4 = nn.Sequential(
            self.MaxPool(3, 1, 1),
            self.Conv(c_in, 64, 1, 1, 1)
        )

    def forward(self, x):
        seq_1 = self.seq_1(x)
        seq_2 = self.seq_2(x)
        seq_3 = self.seq_3(x)
        seq_4 = self.seq_4(x)
        return torch.cat([seq_1, seq_2, seq_3, seq_4], 1)

    def Conv(self, c_in, c_out, kernel_size, padding, dilation=1):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True)
        )

    def MaxPool(self, kernel_size=3, stride=2, padding=0):
        return nn.MaxPool2d(kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer_define = [
            ('c', 3, 64, 3, 1, 1),
            ('c', 64, 64, 3, 1, 1),
            ('p', 3, 2),
            ('c', 64, 128, 3, 1, 1),
            ('c', 128, 128, 3, 1, 1),
            ('p', 3, 2),
            ('c', 128, 256, 3, 1, 1),
            ('c', 256, 256, 3, 1, 1),
            ('c', 256, 256, 3, 1, 1),
            ('p', 3, 2),
            ('c', 256, 512, 3, 1, 1),
            ('c', 512, 512, 3, 1, 1),
            ('c', 512, 512, 3, 1, 1),
            ('p', 3, 1),
            ('c', 512, 512, 3, 1, 2),
            ('c', 512, 512, 3, 1, 2),
            ('c', 512, 512, 3, 1, 2),
            ('i', 512),
            ('i', 512),
        ]
        self.features = self.backbone()
        self.output = nn.Sequential(nn.Conv2d(512, 1, 1, 1), nn.ReLU(inplace=True))
        self.reg = nn.Dropout(p=0.5)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth')
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = self.reg(x)
        return x

    def backbone(self):
        seq = []
        for layer in self.layer_define:
            if layer[0] == 'c':
                seq += self.Conv(*layer[1:])
            elif layer[0] == 'p':
                seq += self.MaxPool(*layer[1:])
            elif layer[0] == 'i':
                seq += [InceptionModule(*layer[1:])]
        return nn.Sequential(*seq)

    def Conv(self, c_in, c_out, kernel_size, padding, dilation):
        return [
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.ReLU6(inplace=True)
        ]

    def MaxPool(self, kernel, stride):
        return [nn.MaxPool2d(kernel, stride)]



if __name__ == "__main__":
    data = torch.zeros(1, 3, 256, 320).cuda()
    m = Model().cuda()
    #print(m)
    a = m(data)
    print(a.shape)