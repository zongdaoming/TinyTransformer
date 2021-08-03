import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseAttention(nn.Module):

    def __init__(self, inplanes, head_count=1):
        super(SiameseAttention, self).__init__()
        self.inplanes = inplanes
        self.head_count = head_count
        self.conv1 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv2 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv3 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv4 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        

    def forward(self, first, second):

        B, C, H, W = first.shape
        N = self.head_count

        first = first.view(B * N, C // N, H, W).contiguous()
        second = second.view(B * N, C // N, H, W).contiguous()
        key = self.conv1(first).view(B * N, C // N, H * W).contiguous()
        # B * N, HW, C / N
        key = key.permute(0, 2, 1).contiguous()
        query = self.conv2(second).view(B * N, C // N, H * W).contiguous()
        # B * N, HW, HW
        res = torch.bmm(key, query)
        res = F.softmax(res, dim=1)
        # B * N, C / N, H, W
        value = self.conv3(first) * self.conv4(second)
        value = value.view(B * N, C // N, H * W)
        # B * N, C / N, H * W
        res = torch.bmm(value, res)
        res = res.view(B, C, H, W)
        return res
        

class SiameseAttentionPlus(nn.Module):

    def __init__(self, inplanes, head_count=1):
        super(SiameseAttentionPlus, self).__init__()
        self.inplanes = inplanes
        self.head_count = head_count
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv2 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv3 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv4 = nn.Conv2d(inplanes // head_count, inplanes // head_count, 1, bias=False)
        self.conv5 = nn.Conv2d(head_count, inplanes, 1, bias=False)

    def forward(self, first, second):

        B, C, H, W = first.shape
        N = self.head_count

        first = first.view(B * N, C // N, H, W).contiguous()
        second = second.view(B * N, C // N, H, W).contiguous()
        key = self.conv1(first).view(B * N, C // N, H * W).contiguous()
        # B * N, HW, C / N
        key = key.permute(0, 2, 1).contiguous()
        query = self.conv2(second).view(B * N, C // N, H * W).contiguous()
        # B * N, HW, HW
        res = torch.bmm(key, query)
        res = F.softmax(res, dim=2)
        # B * N, C / N, H, W
        value1 = self.conv3(first).view(B * N, C // N, H * W).contiguous()
        value2 = self.conv4(second).view(B * N, C // N, H * W).contiguous()
        # B * N, HW, C / N
        value1 = value1.permute(0, 2, 1).contiguous()
        # B * N, HW, HW
        value = torch.bmm(value1, value2)
        # B * N, HW, HW
        res = res * value
        # B * N, HW
        res = torch.sum(res, dim=2)
        res = res.view(B, N, H, W)
        res = self.relu(self.conv5(res))
        return res
        

