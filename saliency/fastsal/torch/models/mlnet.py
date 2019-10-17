'''
Implementation of MLNet model by Cornia et al.

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16


class Model(nn.Module):
    # mlnet model

    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.ModuleList(list(vgg16(pretrained=True).features)[:-1])
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # we create a prior map of 6x8; the original models uses a 3x4 map. We We think this is more effective
        #self.prior = nn.Parameter(torch.ones((1, 1, 6, 8), requires_grad=True))

    def forward(self, inputs):

        features = []
        for idx, feature in enumerate(self.features):
            inputs = feature(inputs)
            if idx in {16, 23, 29}:
                if idx == 16:
                    features.append(F.max_pool2d(inputs, kernel_size=2))
                else:
                    features.append(inputs)

        features = torch.cat((features[0], features[1], features[2]), dim=1)
        output = F.dropout(features, p=0.5)
        output = self.decoder(output)
        #output = output * F.interpolate(self.prior, size=(output.shape[2], output.shape[3]))
        return output
