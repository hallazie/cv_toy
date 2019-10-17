'''
Implementation of various recurrent convolution architectures

@author: Hamed R. Tavakoli
'''


import torch
import torch.nn as nn

import numpy as np


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, spatial_size):
        """

        :param input_channels: number of channels in the input tensor
        :param hidden_channels: number of expected output channels to the hidden space
        :param kernel_size: size of the convolution kernel
        :param spatial_size:
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.spatial_size = spatial_size

        pad_size = int(np.floor(kernel_size/2))
        # define input gate weights
        self.W_xi = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=True)
        self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=False)

        self.W_ci = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define forget gate weights
        self.W_xf = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=True)
        self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=False)

        self.W_cf = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define output gate weights
        self.W_xo = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=True)
        self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=False)

        self.W_co = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define cell weights
        self.W_xc = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=True)
        self.W_hc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=pad_size, bias=False)

    def forward(self, x, h, c):

        ig = self.W_xi(x) + self.W_hi(h) + torch.mul(self.W_ci, c)
        ig = torch.sigmoid(ig)
        fg = self.W_xf(x) + self.W_hf(h) + torch.mul(self.W_cf, c)
        fg = torch.sigmoid(fg)
        c_new = torch.mul(fg, c) + torch.mul(ig, torch.tanh(self.W_xc(x) + self.W_hc(h)))
        output = torch.sigmoid(self.W_xo(x) + self.W_ho(h) + torch.mul(self.W_co, c))
        h_new = torch.mul(output, c_new)
        h_new = h_new.detach()
        c_new = c_new.detach()
        return output, h_new, c_new


# Onle layer convolution LSTM network with one step processing
class ConvLSTM(nn.Module):
    """
        Onle layer convolution LSTM  with one step processing. keeps the states
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        """
        :param input_size: the size of input in the form of [batch size x number of input channels x height x width]
        :param hidden_size:
        :param kernel_size:
        """

        super(ConvLSTM, self).__init__()
        self.Bsize = input_size[0]
        self.input_channel = input_size[1]
        self.Hsize = input_size[2]
        self.Wsize = input_size[3]
        self.hidden_size = hidden_size

        self.cell = ConvLSTMCell(input_channels=self.input_channel, hidden_channels=hidden_size,
                                 kernel_size=kernel_size, spatial_size=[self.Hsize, self.Wsize])
        # initialize hidden state and cell with no prior information to zero
        self.H = torch.zeros((self.Bsize, self.hidden_size, self.Hsize, self.Wsize)).cuda()
        self.C = torch.zeros((self.Bsize, self.hidden_size, self.Hsize, self.Wsize)).cuda()

    def forward(self, x):
        """
            compute one step pass of the convolution LSTM
        :param x: input data
        :return: output x [hidden state x cell state]
        """
        output, self.H, self.C = self.cell(x, self.H, self.C)
        return output, self.H, self.C


if __name__ == "__main__":

    input_x = torch.ones([1, 100, 20, 30]).cuda()
    model = ConvLSTM([1, 100, 20, 30], hidden_size=100, kernel_size=3).cuda()

    for i in range(10000):
        output, _ = model(input_x)
        print(output.shape)
        output = torch.relu(output)
        L = torch.sum(output)
        print(L)
        L.backward(retain_graph=True)
        input_x = output