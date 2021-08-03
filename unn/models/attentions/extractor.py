import torch
import torch.nn as nn
from ..normalize import build_norm_layer

class Single4th(nn.Module):

    def __init__(self, inplanes):
        super(Single4th, self).__init__()

    def forward(self, input):
        return input[4]

class Single3rd(nn.Module):

    def __init__(self, inplanes):
        super(Single3rd, self).__init__()

    def forward(self, input):
        return input[3]

class Fuse4th(nn.Module):

    def __init__(self, inplanes):
        super(Fuse4th, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.norm1_name, norm1 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.norm2_name, norm2 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.norm3_name, norm3 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.norm4_name, norm4 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)
        self.add_module(self.norm4_name, norm4)
        

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)
    
    @property
    def norm4(self):
        return getattr(self, self.norm4_name)

    def forward(self, input):
        feature = input[0]
        feature = self.relu(self.norm1(self.conv1(feature)))
        feature = feature + input[1]
        feature = self.relu(self.norm2(self.conv2(feature)))
        feature = feature + input[2]
        feature = self.relu(self.norm3(self.conv3(feature)))
        feature = feature + input[3]
        feature = self.relu(self.norm4(self.conv4(feature)))
        feature = feature + input[4]
        return feature
              
        
class Fuse3rd(nn.Module):

    def __init__(self, inplanes):
        super(Fuse3rd, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.norm1_name, norm1 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.norm2_name, norm2 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.norm3_name, norm3 = build_norm_layer(inplanes, {'type': 'sync_bn', 'kwargs': {'bn_group_size': 16}}, 1)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)
        

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)
    
    def forward(self, input):
        feature = input[0]
        feature = self.relu(self.norm1(self.conv1(feature)))
        feature = feature + input[1]
        feature = self.relu(self.norm2(self.conv2(feature)))
        feature = feature + input[2]
        feature = self.relu(self.norm3(self.conv3(feature)))
        feature = feature + input[3]
        return feature
       

class conv3x3(nn.Module):

    def __init__(self, inplanes, stride=2):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, 3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(inplanes)
        
    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))


class NewFuse4th(nn.Module):

    def __init__(self, inplanes):
        super(NewFuse4th, self).__init__()
        self.inplanes = inplanes
        self.conv1_1 = conv3x3(inplanes)
        self.conv1_2 = conv3x3(inplanes)
        self.conv1_3 = conv3x3(inplanes)
        self.conv1_4 = conv3x3(inplanes)
        self.conv2_1 = conv3x3(inplanes)
        self.conv2_2 = conv3x3(inplanes)
        self.conv2_3 = conv3x3(inplanes)
        self.conv3_1 = conv3x3(inplanes)
        self.conv3_2 = conv3x3(inplanes)
        self.conv4_1 = conv3x3(inplanes)
        self.conv5_1 = conv3x3(inplanes, 1)
        self.conv6 = conv3x3(inplanes, 1)

    def forward(self, input):
        feature0 = self.conv1_1(input[0])
        feature0 = self.conv1_2(feature0)
        feature0 = self.conv1_3(feature0)
        feature0 = self.conv1_4(feature0)
        feature1 = self.conv2_1(input[1])
        feature1 = self.conv2_2(feature1)
        feature1 = self.conv2_3(feature1)
        feature2 = self.conv3_1(input[2])
        feature2 = self.conv3_2(feature2)
        feature3 = self.conv4_1(input[3])
        feature4 = self.conv5_1(input[4])
        feature = feature0 + feature1 + feature2 + feature3 + feature4
        feature = self.conv6(feature)
        return feature
              
class NewFuse3rd(nn.Module):

    def __init__(self, inplanes):
        super(NewFuse3rd, self).__init__()
        self.inplanes = inplanes
        self.conv1_1 = conv3x3(inplanes)
        self.conv1_2 = conv3x3(inplanes)
        self.conv1_3 = conv3x3(inplanes)
        self.conv2_1 = conv3x3(inplanes)
        self.conv2_2 = conv3x3(inplanes)
        self.conv3_1 = conv3x3(inplanes)
        self.conv4_1 = conv3x3(inplanes, 1)
        self.conv5 = conv3x3(inplanes, 1)

    def forward(self, input):
        feature0 = self.conv1_1(input[0])
        feature0 = self.conv1_2(feature0)
        feature0 = self.conv1_3(feature0)
        feature1 = self.conv2_1(input[1])
        feature1 = self.conv2_2(feature1)
        feature2 = self.conv3_1(input[2])
        feature3 = self.conv4_1(input[3])
        feature = feature0 + feature1 + feature2 + feature3
        feature = self.conv5(feature)
        return feature
              
