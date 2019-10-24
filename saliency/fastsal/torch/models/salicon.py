'''
Implementation of MLNet model by Cornia et al.

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16


class Model(nn.Module):
    # SALICON model

    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(*list(vgg16(pretrained=False).features.children())[:-2])
        self.decoder = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True))

    def forward(self, fine_input):

        coarse_input = F.interpolate(fine_input, scale_factor=0.5)

        fine_input = self.features(fine_input)
        coarse_input = self.features(coarse_input)
        coarse_input = F.interpolate(coarse_input, size=fine_input.shape[2:4])

        fine_input = torch.cat((fine_input, coarse_input), dim=1)
        output = self.decoder(fine_input)
        return output


if __name__ == "__main__":
    data = torch.zeros(1, 3, 600, 800)
    m = Model()
    a = m(data)
    print(a.shape)