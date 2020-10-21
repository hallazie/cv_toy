# --*-- coding:utf-8 --*--

import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)

def run():
    model = Model()
    torch.save(model, 'checkpoint')

def load():
    checkpoint = torch.load('checkpoint')
    state_dict = checkpoint.state_dict()
    # print(dir(checkpoint))
    print(type(checkpoint))
    print(type(state_dict))
    for x in state_dict:
        print(x)
        print(state_dict[x].numpy().shape)
    print('======')
    for x in checkpoint.modules():
        print(x)

if __name__ == '__main__':
    # run()
    load()