'''
Implementation of MLNet model by Cornia et al.

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16


class Model(nn.Module):
    # DeepGazeII model

    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.ModuleList(list(vgg16(pretrained=False).features)[:-1])
        self.decoder = nn.Sequential(
            nn.Conv2d(2560, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, inputs):

        features = []
        for idx, feature in enumerate(self.features):
            inputs = feature(inputs)
            if idx in {24, 25, 27, 28, 29}:
                features.append(inputs)

        features = torch.cat((features[0], features[1], features[2], features[3], features[4]), dim=1)
        output = self.decoder(features)
        return output


if __name__ == "__main__":
    data = torch.zeros(1, 3, 480, 640)
    m = Model()
    a = m(data)
    print(a.shape)