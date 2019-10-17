'''
Implementation of MLNet model by Cornia et al.

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from models.reconv import ConvLSTM


class AttnConvLSTM(nn.Module):
    """
        Attentive Conv LSTM for SAMRES Model
    """
    def __init__(self, input_size, kernel_size, nstep=1):
        """
        :param input_size: the size of input in the form of [batch size x number of input channels x height x width]
        :param kernel_size:
        :param nstep: number of steps to compute the attentive mechanism
        """
        super(AttnConvLSTM, self).__init__()

        attn_size = input_size[1]
        self.bsize = input_size[0]
        self.Hsize = input_size[2]
        self.Wsize = input_size[3]

        self.convLSTM = ConvLSTM(input_size, attn_size, kernel_size)

        self.Wa = nn.Conv2d(input_size[1], input_size[1], kernel_size, padding=int(kernel_size/2), bias=True)
        self.Ua = nn.Conv2d(attn_size, attn_size, kernel_size, padding=int(kernel_size/2), bias=False)
        self.Va = nn.Conv2d(attn_size, 1, kernel_size, padding=int(kernel_size/2), bias=False)

        self.nstep = nstep

    def forward(self, x):

        output, h, c = self.convLSTM(x)
        for i in range(self.nstep):

            Zt = self.Va(torch.tanh(self.Wa(x) + self.Ua(h)))
            At = F.softmax(Zt.view(self.bsize, -1), dim=1).view(self.bsize, 1, self.Hsize, self.Wsize)
            Xt = torch.mul(output, At)
            output, h, c = self.convLSTM(Xt)

        return output


class Model(nn.Module):
    # SAM ResNet model

    def __init__(self, bsize, nstep=3, W=40, H=30):
        """

        :param bsize: batch size
        :param nstep: number of attentionsteps
        :param W: width of expected output
        :param H: height of expected output
        """
        super(Model, self).__init__()

        self.w = W
        self.h = H

        self.features = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.dreduc = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.attn = AttnConvLSTM([bsize, 512, 30, 40], kernel_size=3, nstep=nstep)

        self.decoder0 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=8, dilation=4),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=8, dilation=4),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


    def forward(self, inputs):

        inputs = self.features(inputs)
        inputs = self.dreduc(inputs)
        inputs = F.interpolate(inputs, [self.h, self.w])

        inputs = self.attn(inputs)
        inputs = F.relu(inputs)

        # decoders without learning the prior to keep the number of parameters similar!
        # we remove these in the experiments
        inputs = self.decoder0(inputs)
        inputs = self.decoder1(inputs)

        inputs = self.output(inputs)
        return inputs


if __name__ == "__main__":
    data = torch.ones(1, 3, 240, 320).cuda()
    m = Model(1).cuda()
    output = m(data)
    print(output)
    L = torch.sum(output)
    print(L)
    L.backward()
