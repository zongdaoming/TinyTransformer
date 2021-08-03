import torch.nn as nn
import torch


class PositionalEmbedding(nn.Module):
    def init_param(self, channels, max_h, max_w):
        raise NotImplementedError()

    def __init__(self, channels, max_h=1400, max_w=1400):
        super().__init__()
        if channels >= 256:
            max_h = max_h // 2
            max_w = max_w // 2
        if channels >= 512:
            max_h = max_h // 2
            max_w = max_w // 2
        if channels >= 1024:
            max_h = max_h // 2
            max_w = max_w // 2

        self.h = max_h
        self.ch = max_h // 2
        self.w = max_w
        self.cw = max_w // 2
        self.init_param(channels, max_h, max_w)

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        h_r = h // 2
        w_r = w // 2
        return self.pe[:, :, self.ch - h_r:self.ch - h_r + h, self.cw - w_r:self.cw - w_r + w]


class LearnedEmbedding(PositionalEmbedding):
    def init_param(self, channels, max_h, max_w):
        self.pe = nn.Parameter(torch.zeros(1, channels, max_h, max_w))
